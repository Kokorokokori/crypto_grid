#!/usr/bin/env python3
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset, Order, TradingFee
from lumibot.credentials import IS_BACKTESTING
from lumibot.backtesting import PolygonDataBacktesting

import os
import json
from collections import deque
from statistics import stdev
from datetime import timedelta, datetime
from pathlib import Path

"""
Strategy Description
--------------------
Grid/mean-reversion crypto strategy for BTC/USD using an adaptive envelope based on a rolling standard deviation of the mid price. The bot maintains buy and sell ladders within the band and replaces orders as the market moves, aiming to capture small mean-reversion swings with tight spreads and fees considered.

User Query
----------
This code was refined based on the user prompt: '1. add checks to skip placing a limit order if qty * price < 1 or qty < 0.0001, and round limit_price to the nearest $1 to satisfy the price increment requirement\n2. minimum order size for USD pairs is dynamically computed as 1 ÷ asset price, and for BTC/ETH/USDT pairs it is 1 ÷ (base_asset_price / quote_asset_price). Embedding this formula will make the strategy robust when BTC price swings or if you trade other pairs. Currently the code uses a fixed max_inventory_base but does not compute the broker’s minimum quantity.\n3. change fees: maker/taker schedule should be 0.15% maker and 0.25% taker fees.\n4. Set max_spread_pct to something more realistic (e.g., 1 %).\n5. round each limit_price to the nearest permitted increment and each quantity to a multiple of 0.0001 BTC (or the appropriate increment for other pairs)\n6. Verify that Asset(p["quote_symbol"], asset_type=Asset.AssetType.FOREX) is correct for lumibot/Alpaca’s coin‑pair API\n7. The code divides the available cash by the number of ladders to compute per_buy_notional. If the account balance falls below $6 (with 6 ladders), the calculated notional will be under $1 and orders will be rejected. Add a guard to skip building ladders when per_buy_notional < $1.'
"""


# -------------------- Lightweight CSV Logger (single rolling file) --------------------
class GridTradingLogger:
    _COLUMNS = [
        'ts','event','mid','sigma','spread_pct','equity','dd_pct','daily_loss',
        'cash','base_qty','action','side','qty','price','notional','canceled',
        'buys','sells','reason','extra'
    ]

    def __init__(self, enabled=True, base_dir: Path | None = None):
        self.enabled = bool(enabled)
        self.base_dir = base_dir or self._resolve_logs_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._date = datetime.utcnow().date()

    def _resolve_logs_dir(self) -> Path:
        env_dir = os.getenv('LOG_DIR')
        candidates = []
        if env_dir:
            candidates.append(Path(env_dir))
        candidates.extend([
            Path('/var/data/logs'),  # Render persistent disk
            Path('logs'),            # local relative folder
        ])
        for c in candidates:
            try:
                c.mkdir(parents=True, exist_ok=True)
                # quick write test
                test = c / '.w'
                with open(test, 'w') as f:
                    f.write('1')
                test.unlink(missing_ok=True)
                return c
            except Exception:
                continue
        return Path('.')

    def _ts(self) -> str:
        return datetime.utcnow().isoformat() + 'Z'

    def _file(self) -> Path:
        return self.base_dir / 'grid_trading_log.csv'

    def _rotate_if_needed(self):
        today = datetime.utcnow().date()
        if today != self._date:
            fp = self._file()
            if fp.exists() and fp.stat().st_size > 0:
                archive = self.base_dir / f"grid_trading_log_{self._date.isoformat()}.csv"
                try:
                    fp.rename(archive)
                except Exception:
                    # If rename fails, continue to overwrite header next write
                    pass
            self._date = today

    def log_row(self, row: dict):
        if not self.enabled:
            return
        self._rotate_if_needed()
        fp = self._file()
        write_header = not fp.exists()
        full = {k: row.get(k) for k in self._COLUMNS}
        # Ensure timestamp
        if not full.get('ts'):
            full['ts'] = self._ts()
        import csv
        with open(fp, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self._COLUMNS)
            if write_header:
                w.writeheader()
            w.writerow(full)


class CryptoGridMRStrategy(Strategy):
    # Default parameters can be overridden by providing parameters on init or via set_parameters()
    parameters = {
        # Trading pair
        "base_symbol": "BTC",
        "quote_symbol": "USD",

        # Grid settings
        "ladders": 6,                  # N buy + N sell rungs
        "step_pct": 0.004,             # 0.4% between rungs
        "envelope_k_sigma": 2.0,       # bands = mid ± k * sigma
        "sigma_window_sec": 60,        # rolling window for sigma (live default)
        "sigma_window_backtest_minutes": 5,  # use 5-minute window in backtests for robust sigma
        "rebuild_interval_sec": 5,     # how often to rebuild ladder (live); adapted in backtests
        "max_spread_pct": 0.01,        # 1% spread safety
        "max_slippage_pct": 0.10,      # informational (orders are limit)

    # Fees (maker/taker) - updated per request
        "maker_fee_pct": 0.0015,       # 0.15%
        "taker_fee_pct": 0.0025,       # 0.25%

        # Risk limits
        "max_inventory_quote": 2000.0,   # cap net USD exposure (buy-side)
        "max_inventory_base": 0.2,       # cap net BTC inventory (sell-side)
        "max_drawdown_pct": 10.0,        # kill if equity drawdown exceeds this
        "daily_loss_limit_quote": 300.0, # kill if daily loss exceeds this USD amount
    "kill_switch_on_ws_stall_sec": 20,  # stale data threshold (live only) - raised to reduce spurious kills
        "flatten_on_kill": True,          # close positions when killed

        # Rounding/increment controls (can be overridden per exchange/pair)
        # If None, sensible defaults are inferred from symbols
        "price_increment": None,        # e.g., 1.0 for USD pairs per request
        "qty_increment": None,          # e.g., 0.0001 for BTC

    # Profitability guard: require step to exceed fees by a margin
    "min_edge_pct": 0.0005,        # 5 bps additional edge on top of 2x maker fee

        # Control switch
        "enabled": True,
        # Auto-clear a stale kill flag (e.g., from a prior crash) when enabled=True
        # This avoids manual toggling after harmless startup issues.
        "auto_clear_kill": True,

        # Stall kill debounce: consecutive stall events required before kill
        "stall_kill_strikes": 3,

        # Ladder and management controls
        # Guarantee at least this many step widths inside the envelope (each side can host up to `ladders` rungs)
        "min_rungs_in_band": 5,
        # Tolerance to keep existing orders near targets (in whole price ticks)
        "target_tolerance_ticks": 2,
    }

    def initialize(self):
        # Crypto trades 24/7; this keeps the bot running around the clock
        self.set_market("24/7")

        # Run frequent iterations by default (live); in backtests we align to minute bars below
        self.sleeptime = "1S"

        # Build asset objects once (store in self.vars as required by framework guidelines)
        p = self.get_parameters()
        self.base_asset = Asset(p["base_symbol"], asset_type=Asset.AssetType.CRYPTO)

        # IMPORTANT: For crypto orders/quotes, use CRYPTO for the quote asset as well
        self.quote_asset_for_orders = Asset(p["quote_symbol"], asset_type=Asset.AssetType.CRYPTO)

        # Rolling mid-price window (timestamped) for sigma calculation
        # IMPORTANT: Keep this out of self.vars because Lumibot persists self.vars to JSON
        # and collections.deque is not JSON-serializable.
        self.mid_window = deque()  # elements: (timestamp, mid)

        # State and persistence
        self.vars.last_rebuild_ts = None
        self.vars.killed = False
        self.vars.flatten_executed = False
        self.vars.ladder_prices = {"buys": [], "sells": []}  # last planned ladder levels
        self.vars.last_mid = None
        self.vars.last_sigma = None
        self.vars.last_quote_ts = None
        self.vars.needs_initial_reconcile = True
        # Stall debounce counter
        self.vars.stall_strikes = 0

        # Risk tracking
        current_equity = self.get_portfolio_value() or 0.0
        self.vars.daily_start_equity = current_equity
        self.vars.equity_highwater = current_equity
        self.vars.daily_date = self.get_datetime().date() if self.get_datetime() else None

        # Effective rebuild cadence & sigma window (make backtests align to minute bars)
        self.vars.rebuild_interval_effective = int(p["rebuild_interval_sec"]) if int(p["rebuild_interval_sec"]) > 0 else 5
        # Use an effective sigma window value so we can adjust for backtesting without mutating provided parameters
        self.vars.sigma_window_effective_sec = int(p.get("sigma_window_sec", 60))

        if self.is_backtesting:
            # Align to minute bars in backtests so each iteration processes a completed bar
            self.sleeptime = "1M"  # 1 minute cadence for backtests
            self.vars.rebuild_interval_effective = 60  # rebuild once per bar
            # Use a multi-minute window so we accumulate enough samples for stdev on minute data
            minutes = int(p.get("sigma_window_backtest_minutes", 5))
            if minutes <= 0:
                minutes = 5
            self.vars.sigma_window_effective_sec = 60 * minutes
            self.log_message(
                f"Backtest mode: sleeptime=1M, rebuild_interval=60s, sigma_window={self.vars.sigma_window_effective_sec}s.",
                color="yellow",
            )

        # Persistence file (simple JSON to make restart safe)
        # IMPORTANT: Do NOT store the path in self.vars because the Lumibot DB JSON loader
        # tries to parse any string containing 'T' as an ISO timestamp. 'BTCUSD' contains 'T'.
        # Keep this as a normal attribute to avoid repeated load errors.
        self.state_path = f"grid_{p['base_symbol']}{p['quote_symbol']}_state.json"
        # Proactively remove any lingering value from prior runs
        try:
            if hasattr(self.vars, 'state_path'):
                setattr(self.vars, 'state_path', None)
        except Exception:
            pass
        # Initialize CSV logger (works on Render and locally)
        self.grid_logger = GridTradingLogger(enabled=True)

        self._load_state()
        # Optionally clear a persisted kill from prior runs if the user has enabled trading
        if p.get("auto_clear_kill", True) and p.get("enabled", True) and self.vars.killed:
            self.vars.killed = False
            self.vars.flatten_executed = False
            self.log_message("Kill switch cleared at startup (auto_clear_kill=True, enabled=True).", color="green")
        self.log_message("Initialized crypto grid strategy with 24/7 market.", color="green")
        # Log init
        try:
            self.grid_logger.log_row({'event': 'INIT'})
        except Exception:
            pass

    # -------------------- Utility: Persistence --------------------
    def _load_state(self):
        path = getattr(self, 'state_path', None)
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                # Restore core items; if killed in prior run, keep it killed
                self.vars.killed = bool(data.get("killed", False))
                self.vars.last_mid = data.get("last_mid")
                self.vars.last_sigma = data.get("last_sigma")
                self.vars.ladder_prices = data.get("ladder_prices", {"buys": [], "sells": []})
                self.vars.equity_highwater = data.get("equity_highwater", self.vars.equity_highwater)
                self.vars.daily_start_equity = data.get("daily_start_equity", self.vars.daily_start_equity)
                self.log_message("State loaded from disk; will reconcile on first iteration.", color="yellow")
            except Exception as e:
                self.log_message(f"Failed to load state: {e}", color="red")

    def _save_state(self):
        data = {
            "killed": self.vars.killed,
            "last_mid": self.vars.last_mid,
            "last_sigma": self.vars.last_sigma,
            "ladder_prices": self.vars.ladder_prices,
            "equity_highwater": self.vars.equity_highwater,
            "daily_start_equity": self.vars.daily_start_equity,
            "timestamp": self.get_timestamp(),
        }
        try:
            path = getattr(self, 'state_path', None)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            self.log_message(f"Failed to save state: {e}", color="red")

    # -------------------- Utility: Increments & minimums --------------------
    def _get_price_increment(self) -> float:
        # Allow explicit override via parameters
        p = self.get_parameters()
        if p.get("price_increment") is not None:
            return float(p["price_increment"])
        quote = p["quote_symbol"].upper()
        # Defaults: USD pairs round to $1, stables to $0.01; BTC/ETH quotes tighter
        if quote == "USD":
            return 1.0
        if quote in ("USDT", "USDC"):
            return 0.01
        if quote == "BTC":
            return 0.000001
        if quote == "ETH":
            return 0.00001
        return 0.01

    def _get_qty_increment(self) -> float:
        p = self.get_parameters()
        if p.get("qty_increment") is not None:
            return float(p["qty_increment"])
        base = p["base_symbol"].upper()
        # Sensible defaults; user specifically asked for 0.0001 BTC step
        if base == "BTC":
            return 0.0001
        if base == "ETH":
            return 0.001
        # Default to a fine step for other coins
        return 0.0001

    def _round_to_increment(self, value: float, increment: float) -> float:
        if value is None or increment is None or increment <= 0:
            return value
        return round(value / increment) * increment

    def _floor_to_increment(self, value: float, increment: float) -> float:
        if value is None or increment is None or increment <= 0:
            return value
        scaled = int(value / increment)
        return scaled * increment

    def _ceil_to_increment(self, value: float, increment: float) -> float:
        if value is None or increment is None or increment <= 0:
            return value
        # Use floor on the negated value to derive ceil without introducing new imports
        return -self._floor_to_increment(-value, increment)

    def _compute_min_qty(self, mid_in_quote: float = None) -> float:
        """
        Minimum order size as 1 unit of quote currency: min_qty = 1 / price_in_quote.
        Requires a valid price passed by the caller; no implicit price lookup.
        """
        if mid_in_quote is None or mid_in_quote <= 0:
            return None
        price_in_quote = float(mid_in_quote)
        return 1.0 / price_in_quote

    # -------------------- Utility: Risk & Kill Switch --------------------
    def _check_daily_reset(self):
        # Reset daily loss counter at UTC midnight
        now_dt = self.get_datetime()
        if not now_dt:
            return
        if self.vars.daily_date is None:
            self.vars.daily_date = now_dt.date()
            return
        if now_dt.date() != self.vars.daily_date:
            self.vars.daily_date = now_dt.date()
            self.vars.daily_start_equity = self.get_portfolio_value() or 0.0
            self.log_message("Daily reset applied; new daily start equity set.", color="blue")

    def _risk_checks(self):
        p = self.get_parameters()
        if not p.get("enabled", True):
            return

        equity = self.get_portfolio_value() or 0.0
        if equity > (self.vars.equity_highwater or 0):
            self.vars.equity_highwater = equity
        if (self.vars.equity_highwater or 0) > 0:
            dd = (self.vars.equity_highwater - equity) / self.vars.equity_highwater
        else:
            dd = 0.0
        daily_loss = (self.vars.daily_start_equity or 0.0) - equity

        self.log_message(
            f"Risk check: equity={equity:.2f}, dd={dd*100:.2f}%, daily_loss={daily_loss:.2f}",
            color="blue",
        )

        if dd >= (p["max_drawdown_pct"] / 100.0):
            self._activate_kill_switch("Max drawdown breached")
            return
        if daily_loss >= p["daily_loss_limit_quote"]:
            self._activate_kill_switch("Daily loss limit breached")
            return
        # Log risk snapshot
        try:
            self.grid_logger.log_row({
                'event': 'RISK',
                'equity': equity,
                'dd_pct': dd * 100.0,
                'daily_loss': daily_loss,
            })
        except Exception:
            pass

    def _activate_kill_switch(self, reason: str):
        if self.vars.killed:
            return
        self.vars.killed = True
        self.log_message(f"KILL SWITCH ACTIVATED: {reason}", color="red")
        self.cancel_open_orders()
        if self.get_parameters().get("flatten_on_kill", True):
            self._flatten_positions()
            self.vars.flatten_executed = True
        self._save_state()

    def _flatten_positions(self):
        # Close positions to get back to cash. For crypto, we simply sell what we hold (no shorts assumed).
        positions = self.get_positions()
        for pos in positions:
            if pos.asset.asset_type == Asset.AssetType.FOREX and pos.asset.symbol == "USD":
                continue
            if pos.asset.asset_type == Asset.AssetType.CRYPTO and pos.asset.symbol == self.base_asset.symbol:
                qty = pos.quantity
                if qty is None or qty == 0:
                    continue
                side = Order.OrderSide.SELL if qty > 0 else Order.OrderSide.BUY
                order = self.create_order(
                    pos.asset,
                    abs(qty),
                    side,
                    order_type=Order.OrderType.MARKET,
                    quote=self.quote_asset_for_orders,
                )
                self.submit_order(order)
                self.log_message(
                    f"Flatten order submitted: {side} {abs(qty)} {pos.asset.symbol} (market)",
                    color="yellow",
                )

    # -------------------- Utility: Market Data & Stats --------------------
    def _get_mid_and_spread(self):
        # Try to get a fresh quote with bid/ask; do not fallback to last trade price
        q = self.get_quote(self.base_asset, quote=self.quote_asset_for_orders)
        now_dt = self.get_datetime()
        mid = None
        spread_pct = None
        if q is not None:
            if q.bid is not None and q.ask is not None and q.bid > 0 and q.ask > 0 and q.ask >= q.bid:
                mid = (q.bid + q.ask) / 2.0
                if q.ask > 0 and mid and mid > 0:
                    spread_pct = (q.ask - q.bid) / mid
                # Consider this a fresh data point and record local receipt time
                self.vars.last_quote_ts = now_dt
        return mid, spread_pct

    def _update_sigma_window(self, mid):
        # Keep only points within the effective sigma window and compute percent-volatility
        now_dt = self.get_datetime()
        window_sec = int(self.vars.sigma_window_effective_sec)
        if not now_dt or mid is None:
            print(f"Sigma early exit: now_dt={now_dt}, mid={mid}")
            return None
        self.mid_window.append((now_dt, float(mid)))
        cutoff = now_dt - timedelta(seconds=window_sec)
        while self.mid_window and self.mid_window[0][0] < cutoff:
            self.mid_window.popleft()

        values = [v for (_, v) in self.mid_window]
        print(f"Sigma debug: window_sec={window_sec}, values_count={len(values)}")
        if len(values) < 2 or max(values) <= 0:
            print(f"Not enough values: {len(values)} < 2")
            return None

        # Compute simple returns between consecutive mids to derive percent volatility
        try:
            returns = []
            for i in range(1, len(values)):
                prev = values[i - 1]
                curr = values[i]
                if prev and prev > 0:
                    returns.append((curr / prev) - 1.0)
            # Require fewer samples in backtests (minute cadence) than in live (second cadence)
            min_required = 5 if self.is_backtesting else 10
            if len(returns) >= min_required:
                s_pct = stdev(returns)
                print(f"Sigma SUCCESS (pct): {s_pct}")
                return s_pct
            else:
                print(f"Not enough returns: {len(returns)} < {min_required}")
                return None
        except Exception as e:
            print(f"Sigma calculation error: {e}")
            return None

    # -------------------- Helper: open orders filter (selective management) --------------------
    def _get_open_orders_for_base_asset(self):
        # Filter open/active orders for our base asset symbol
        open_statuses = {
            Order.OrderStatus.OPEN,
            Order.OrderStatus.SUBMITTED,
            Order.OrderStatus.NEW,
        }
        # Also check for string status values (in case of broker differences)
        open_status_strings = {"open", "submitted", "new", "pending_new", "accepted"}
        
        active = []
        for o in self.get_orders() or []:
            if o is None or o.asset is None:
                continue
            if o.asset.asset_type != Asset.AssetType.CRYPTO:
                continue
            if o.asset.symbol != self.base_asset.symbol:
                continue
            
            order_status = getattr(o, "status", None)
            # Check both enum and string status formats
            if order_status in open_statuses or str(order_status).lower() in open_status_strings:
                active.append(o)
        return active

    # -------------------- Ladder Building & Order Placement --------------------
    def _enforced_step_pct(self):
        p = self.get_parameters()
        min_step = 2.0 * p["maker_fee_pct"] + float(p["min_edge_pct"])
        step = p["step_pct"]
        if step < min_step:
            self.log_message(
                f"Step pct {step:.4f} raised to {min_step:.4f} to exceed fees + edge.",
                color="yellow",
            )
            return min_step
        return step

    def _rebuild_ladder(self, mid, sigma):
        p = self.get_parameters()
        if self.vars.killed or not p.get("enabled", True):
            self.log_message("Trading disabled or killed; skip ladder rebuild.", color="yellow")
            return

        if mid is None or mid <= 0:
            self.log_message("No valid mid price; skip ladder rebuild.", color="red")
            return

        # Calculate band
        step_pct = self._enforced_step_pct()
        if sigma is None or sigma <= 0:
            self.log_message("Sigma not ready; skipping ladder rebuild this iteration.", color="yellow")
            return
        else:
            # sigma is percent volatility (stdev of returns). Convert to price band around mid.
            k = p["envelope_k_sigma"]
            sigma_pct = float(sigma)
            # Ensure at least N step widths fit inside the envelope to host multiple rungs
            min_rungs = int(p["min_rungs_in_band"]) if int(p["min_rungs_in_band"]) > 0 else 1
            band_pct = max(k * sigma_pct, step_pct * min_rungs)
            lower = max(1e-9, mid * (1.0 - band_pct))
            upper = mid * (1.0 + band_pct)

        # Visualize key reference lines (keep charts clear)
        self.add_line("BTC Mid", float(mid), color="black", width=2)
        self.add_line("Upper Band", float(upper), color="blue", width=1)
        self.add_line("Lower Band", float(lower), color="blue", width=1)

        # Inventory checks: use current holdings to size orders conservatively
        base_pos = self.get_position(self.base_asset)
        base_qty = float(base_pos.quantity) if base_pos is not None and base_pos.quantity is not None else 0.0
        cash_quote = float(self.get_cash() or 0.0)

        # Caps
        max_quote = float(p["max_inventory_quote"])  # for buys
        max_base = float(p["max_inventory_base"])    # for sells (reduce-only)

        # Available for NEW buys: do not exceed cash and quote cap
        buy_notional_cap = min(cash_quote, max_quote)
        # Available for NEW sells: only sell what we have, but keep some buffer
        sell_qty_cap = max(0.0, min(base_qty, max_base))

        # Per-rung sizing: spread risk evenly across ladders
        ladders = int(p["ladders"]) if p["ladders"] > 0 else 1
        per_buy_notional = buy_notional_cap / ladders if ladders > 0 else 0.0
        per_sell_qty = sell_qty_cap / ladders if ladders > 0 else 0.0

        # Debug: Show detailed inventory and sizing calculations
        self.log_message(f"Ladder rebuild: Mid={mid:.2f}, Sigma={sigma}, Step={step_pct:.4f}", color="cyan")
        self.log_message(f"Inventory: BTC={base_qty:.6f}, Cash=${cash_quote:.2f}", color="cyan")
        print(f"LADDER DEBUG: Mid={mid:.2f}, Sigma={sigma}, Cash=${cash_quote:.2f}, BTC={base_qty:.6f}")
        print(f"LADDER DEBUG: per_buy_notional=${per_buy_notional:.2f}, per_sell_qty={per_sell_qty:.6f}")

        # Guard: if per_buy_notional < $1, skip building buy ladders to avoid rejected orders
        if per_buy_notional < 1.0:
            print(f"WARNING: Per-rung buy notional below $1 ({per_buy_notional:.2f}); skipping buy ladder this cycle.")
            self.log_message(
                f"Per-rung buy notional below $1 ({per_buy_notional:.2f}); skipping buy ladder this cycle.",
                color="yellow",
            )
        else:
            print(f"Per-buy notional OK: ${per_buy_notional:.2f}")

        # Pre-compute increments and dynamic minimum qty
        price_inc = self._get_price_increment()
        qty_inc = self._get_qty_increment()
        dyn_min_qty = self._compute_min_qty(mid)
        if dyn_min_qty is None:
            self.log_message("Min qty unavailable (no price); skipping ladder rebuild.", color="yellow")
            return
        hard_min_qty = 0.0001  # universal guard

        # Generate target prices inside the envelope and round to allowed tick
        target_buys = []
        target_sells = []
        for i in range(1, ladders + 1):  # i starts at 1 to skip mid
            bp_raw = mid * (1.0 - step_pct * i)
            sp_raw = mid * (1.0 + step_pct * i)
            if bp_raw >= lower:
                # Directional rounding to make resting prices sticky for buys
                target_buys.append(self._floor_to_increment(float(bp_raw), price_inc))
            if sp_raw <= upper:
                # Directional rounding to make resting prices sticky for sells
                target_sells.append(self._ceil_to_increment(float(sp_raw), price_inc))

        target_buy_set = set([tp for tp in target_buys if tp > 0])
        target_sell_set = set([tp for tp in target_sells if tp > 0])

        # Log target ladder information
        self.log_message(f"Targets: {len(target_buys)} buys, {len(target_sells)} sells", color="cyan")
        self.log_message(f"Per-buy notional: ${per_buy_notional:.2f}, Per-sell qty: {per_sell_qty:.6f}", color="cyan")

        # SELECTIVE ORDER MANAGEMENT
        # 1) Inspect existing open orders and cancel only those that are stale or outside envelope
        existing_open = self._get_open_orders_for_base_asset()
        all_orders = self.get_orders() or []
        print(f"ORDER DEBUG: Total orders={len(all_orders)}, BTC orders={len(existing_open)}")
        if all_orders:
            for i, o in enumerate(all_orders[:3]):  # Show first 3 orders
                if o and hasattr(o, 'asset') and o.asset:
                    print(f"  Order {i}: {o.asset.symbol} {getattr(o, 'side', '?')} {getattr(o, 'status', '?')}")
        self.log_message(f"Found {len(existing_open)} existing open orders", color="cyan")
        to_cancel = []
        kept_buy_prices = set()
        kept_sell_prices = set()

        # Use explicit multi-tick tolerance to avoid cancel/repost churn on tiny mid moves
        tolerance_ticks = int(p["target_tolerance_ticks"]) if int(p["target_tolerance_ticks"]) >= 0 else 0
        tol = (tolerance_ticks * price_inc) if price_inc > 0 else 0.0

        for o in existing_open:
            if o.limit_price is None:
                to_cancel.append(o)
                continue
            lp = float(o.limit_price)
            # Cancel if order is outside current envelope
            if lp < lower or lp > upper:
                to_cancel.append(o)
                continue

            # Keep orders that are inside the envelope; only cancel if far from any target beyond tolerance
            if o.side == Order.OrderSide.BUY:
                # If no targets computed, keep inside-envelope order as-is
                if not target_buy_set:
                    kept_buy_prices.add(lp)
                    continue
                nearest = min(target_buy_set, key=lambda x: abs(x - lp))
                if abs(nearest - lp) <= tol:
                    kept_buy_prices.add(nearest)
                else:
                    to_cancel.append(o)
            elif o.side == Order.OrderSide.SELL:
                if not target_sell_set:
                    kept_sell_prices.add(lp)
                    continue
                nearest = min(target_sell_set, key=lambda x: abs(x - lp))
                if abs(nearest - lp) <= tol:
                    kept_sell_prices.add(nearest)
                else:
                    to_cancel.append(o)

        # Execute cancellations
        if to_cancel:
            for co in to_cancel:
                self.cancel_order(co)
            self.log_message(f"Canceled {len(to_cancel)} stale/out-of-envelope orders.", color="yellow")
            try:
                self.grid_logger.log_row({'event': 'CANCEL', 'canceled': len(to_cancel)})
            except Exception:
                pass
            
            # Add small delay to ensure cancellations are processed before placing new orders
            # This prevents potential ID conflicts between canceled and new orders
            import time
            time.sleep(0.1)  # 100ms delay

        # 2) Place NEW orders only for targets not already covered by kept orders
        self.vars.ladder_prices["buys"], self.vars.ladder_prices["sells"] = [], []
        buys_placed = 0
        sells_placed = 0

        # BUY side
        missing_buy_targets = sorted([p for p in target_buy_set if p not in kept_buy_prices])
        if per_buy_notional >= 1.0:
            for limit_price in missing_buy_targets:
                if limit_price <= 0:
                    continue
                qty_raw = max(0.0, per_buy_notional / limit_price)
                qty = self._round_to_increment(qty_raw, qty_inc)
                if qty < dyn_min_qty:
                    qty = self._round_to_increment(dyn_min_qty, qty_inc)
                if qty * limit_price < 1.0 or qty < hard_min_qty:
                    self.log_message(
                        f"Skip NEW BUY @ {limit_price:.2f}: qty={qty:.6f} fails min ($1 or 0.0001).",
                        color="yellow",
                    )
                    continue
                order = self.create_order(
                    self.base_asset,
                    qty,
                    Order.OrderSide.BUY,
                    order_type=Order.OrderType.LIMIT,
                    limit_price=float(limit_price),
                    quote=self.quote_asset_for_orders,
                )
                self.submit_order(order)
                buys_placed += 1
                print(f"NEW BUY ORDER SUBMITTED: {qty:.6f} @ ${limit_price:.2f} (notional=${qty*limit_price:.2f})")
                # Avoid emoji in logs to prevent Windows console encoding errors
                self.log_message(f"NEW BUY SUBMITTED: {qty:.6f} @ ${limit_price:.2f} (notional=${qty*limit_price:.2f})", color="green")
                try:
                    self.grid_logger.log_row({
                        'event': 'NEW_ORDER', 'action': 'SUBMIT', 'side': 'BUY',
                        'qty': qty, 'price': float(limit_price),
                        'notional': float(qty*limit_price)
                    })
                except Exception:
                    pass
        else:
            self.log_message("Buy ladder skipped due to insufficient per-rung notional.", color="yellow")

        # SELL side (reduce-only)
        missing_sell_targets = sorted([p for p in target_sell_set if p not in kept_sell_prices])
        for limit_price in missing_sell_targets:
            if per_sell_qty <= 0.0:
                self.log_message("No base inventory to place NEW sells (reduce-only).", color="yellow")
                break
            if limit_price <= 0:
                continue
            qty = self._round_to_increment(per_sell_qty, qty_inc)
            if qty < dyn_min_qty:
                qty = self._round_to_increment(dyn_min_qty, qty_inc)
            if qty * limit_price < 1.0 or qty < hard_min_qty:
                self.log_message(
                    f"Skip NEW SELL @ {limit_price:.2f}: qty={qty:.6f} fails min ($1 or 0.0001).",
                    color="yellow",
                )
                continue
            order = self.create_order(
                self.base_asset,
                qty,
                Order.OrderSide.SELL,
                order_type=Order.OrderType.LIMIT,
                limit_price=float(limit_price),
                time_in_force="gtc",
                quote=self.quote_asset_for_orders,
            )
            self.submit_order(order)
            sells_placed += 1
            self.log_message(f"Placed NEW SELL limit {qty:.6f} @ {limit_price:.2f}", color="green")
            try:
                self.grid_logger.log_row({
                    'event': 'NEW_ORDER', 'action': 'SUBMIT', 'side': 'SELL',
                    'qty': qty, 'price': float(limit_price),
                    'notional': float(qty*limit_price)
                })
            except Exception:
                pass

        # Refresh ladder prices from currently open orders so state reflects what is active now
        self.vars.ladder_prices["buys"], self.vars.ladder_prices["sells"] = [], []
        for o in self._get_open_orders_for_base_asset():
            if o.side == Order.OrderSide.BUY:
                self.vars.ladder_prices["buys"].append(float(o.limit_price))
            elif o.side == Order.OrderSide.SELL:
                self.vars.ladder_prices["sells"].append(float(o.limit_price))

        self.log_message(
            f"Ladder rebuild complete: new buys={buys_placed}, new sells={sells_placed}, canceled={len(to_cancel)}.",
            color="blue",
        )
        try:
            self.grid_logger.log_row({
                'event': 'LADDER', 'mid': float(mid), 'sigma': float(sigma),
                'cash': cash_quote, 'base_qty': base_qty,
                'buys': buys_placed, 'sells': sells_placed, 'canceled': len(to_cancel)
            })
        except Exception:
            pass
        
        # Debug: Show open BTC orders after rebuild (exclude canceled/filled for clarity)
        open_btc_orders = [o for o in (self._get_open_orders_for_base_asset() or [])]
        if open_btc_orders:
            self.log_message(f"Total BTC OPEN orders after rebuild: {len(open_btc_orders)}", color="cyan")
            for o in open_btc_orders[:3]:  # Show first 3 open orders
                status = getattr(o, 'status', 'unknown')
                self.log_message(f"  Order: {o.side} {getattr(o, 'quantity', '?')} @ ${getattr(o, 'limit_price', '?')} [{status}]", color="white")

        self.vars.last_rebuild_ts = self.get_timestamp()
        self.vars.last_mid = float(mid)
        self.vars.last_sigma = float(sigma) if sigma is not None else None
        self._save_state()

    # -------------------- Lifecycle: Core Loop --------------------
    def on_trading_iteration(self):
        print(f"ITERATION START - Portfolio: ${self.get_portfolio_value():.2f}")
        p = self.get_parameters()

        # If a stale kill flag was persisted, optionally clear it automatically when enabled=True
        if p.get("auto_clear_kill", True) and p.get("enabled", True) and self.vars.killed:
            self.vars.killed = False
            self.vars.flatten_executed = False
            self.log_message("Kill switch cleared automatically before iteration (auto_clear_kill=True).", color="green")

        # Respect on/off switch
        if not p.get("enabled", True):
            print("Trading disabled")
            self.log_message("Trading is disabled via parameters; standing by.", color="yellow")
            return

        # If killed previously, keep idle (user can re-enable with parameters)
        if self.vars.killed:
            print("Kill switch active")
            if p.get("flatten_on_kill", True) and not self.vars.flatten_executed:
                self._flatten_positions()
                self.vars.flatten_executed = True
            self.log_message("Kill switch is active; trading paused until re-enabled.", color="red")
            return

        # On first iteration after (re)start, reconcile by canceling stray orders
        if self.vars.needs_initial_reconcile:
            self.cancel_open_orders()
            self.vars.needs_initial_reconcile = False
            self.log_message("Initial reconcile complete: all stray orders canceled.", color="blue")

        # Daily reset and risk checks
        self._check_daily_reset()
        self._risk_checks()
        if self.vars.killed:
            return

        # Market data: get mid & spread
        mid, spread_pct = self._get_mid_and_spread()
        print(f"Market data: Mid=${mid}, Spread={spread_pct}")
        
        if mid is None:
            print("No price data available")
            self.log_message("No price available this iteration; waiting for data.", color="yellow")
            return

        # Apply stall kill-switch only in live trading
        now_dt = self.get_datetime()
        print(f"Checking stall kill-switch: is_backtesting={self.is_backtesting}")
        if (not self.is_backtesting) and self.vars.last_quote_ts and now_dt:
            stall = (now_dt - self.vars.last_quote_ts).total_seconds()
            if stall > p["kill_switch_on_ws_stall_sec"]:
                # Increment stall strikes and only kill after N consecutive stalls
                self.vars.stall_strikes = int(getattr(self.vars, 'stall_strikes', 0) or 0) + 1
                strikes_needed = int(p.get("stall_kill_strikes", 3))
                print(f"STALL DETECTED: {stall}s > {p['kill_switch_on_ws_stall_sec']}s (strike {self.vars.stall_strikes}/{strikes_needed})")
                if self.vars.stall_strikes >= strikes_needed:
                    print(f"STALL KILL ACTIVATED after {self.vars.stall_strikes} strikes")
                    self._activate_kill_switch("Data feed stale beyond threshold")
                    self.vars.stall_strikes = 0
                    return
            else:
                # Fresh enough data; reset strikes
                if getattr(self.vars, 'stall_strikes', 0):
                    print("STALL RESET: data fresh, strikes cleared")
                self.vars.stall_strikes = 0

        # Spread safety check (if spread known)
        print(f"Checking spread: {spread_pct}")
        if spread_pct is not None and spread_pct > p["max_spread_pct"]:
            print(f"SPREAD TOO WIDE: {spread_pct*100:.2f}% > {p['max_spread_pct']*100:.2f}%")
            self.log_message(
                f"Spread too wide ({spread_pct*100:.2f}% > {p['max_spread_pct']*100:.2f}%); skipping.",
                color="yellow",
            )
            return

        # Update sigma with the new mid price
        print(f"Updating sigma with mid={mid}")
        sigma = self._update_sigma_window(mid)
        print(f"Sigma calculated: {sigma}")
        if sigma is None:
            # No fallback envelope: skip trading until we have enough samples
            print("Sigma not ready - skipping this iteration")
            self.log_message("Sigma not ready (need more samples); skipping this iteration.", color="yellow")
            return

        # Rebuild ladder only every effective interval (60s in backtests to match minute bars)
        ts = self.get_timestamp()
        interval_sec = self.vars.rebuild_interval_effective
        if self.vars.last_rebuild_ts is None or (ts - self.vars.last_rebuild_ts) >= interval_sec:
            print(f"REBUILDING LADDER: Mid=${mid:.2f}, Sigma={sigma}")
            self.log_message(f"REBUILDING LADDER: Mid=${mid:.2f}, Sigma={sigma}", color="yellow")
            self._rebuild_ladder(mid, sigma)
        else:
            time_until_rebuild = interval_sec - (ts - self.vars.last_rebuild_ts)
            # Don't spam the logs with rebuild countdown
            if time_until_rebuild > 50:  # Only log once near the beginning
                print(f"Next rebuild in {time_until_rebuild:.0f}s")

    # -------------------- Fills: Re-quote the counterpart rung --------------------
    def on_filled_order(self, position, order, price, quantity, multiplier):
        self.log_message(f"FILL: {order.side} {quantity} @ {price}", color="green")
        try:
            self.grid_logger.log_row({'event': 'FILL', 'side': str(order.side), 'qty': float(quantity), 'price': float(price)})
        except Exception:
            pass
        
        # When a buy fills, place a matching sell above; when a sell fills, place a buy below
        if order is None or order.asset is None:
            return
        if order.asset.asset_type != Asset.AssetType.CRYPTO or order.asset.symbol != self.base_asset.symbol:
            return

        step_pct = self._enforced_step_pct()
        filled_px = float(price) if price is not None else None
        filled_qty = float(quantity) if quantity is not None else None
        if filled_px is None or filled_qty is None or filled_px <= 0 or filled_qty <= 0:
            return

        p = self.get_parameters()
        if self.vars.killed or not p.get("enabled", True):
            return

        # Apply increment and minimum rules to post-fill re-quote
        price_inc = self._get_price_increment()
        qty_inc = self._get_qty_increment()
        hard_min_qty = 0.0001

        side = order.side
        if side == Order.OrderSide.BUY:
            target_px_raw = filled_px * (1.0 + step_pct)
            target_px = self._round_to_increment(target_px_raw, price_inc)
            qty = self._round_to_increment(filled_qty, qty_inc)
            dyn_min_qty = self._compute_min_qty(target_px)
            if dyn_min_qty is None:
                self.log_message("Post-fill SELL skipped: min qty unavailable.", color="yellow")
                return
            if qty < dyn_min_qty:
                qty = self._round_to_increment(dyn_min_qty, qty_inc)
            if qty * target_px < 1.0 or qty < hard_min_qty:
                self.log_message(
                    f"Post-fill SELL skipped: qty={qty:.6f}, price={target_px:.2f} fails min.",
                    color="yellow",
                )
            else:
                order_out = self.create_order(
                    self.base_asset,
                    qty,
                    Order.OrderSide.SELL,
                    order_type=Order.OrderType.LIMIT,
                    limit_price=float(target_px),
                    time_in_force="gtc",
                    quote=self.quote_asset_for_orders,
                )
                self.submit_order(order_out)
                self.log_message(f"Post-fill SELL placed {qty:.6f} @ {target_px:.2f}", color="green")
                try:
                    self.grid_logger.log_row({'event': 'NEW_ORDER', 'action': 'POST_FILL', 'side': 'SELL', 'qty': qty, 'price': float(target_px)})
                except Exception:
                    pass
        elif side == Order.OrderSide.SELL:
            target_px_raw = filled_px * (1.0 - step_pct)
            target_px = self._round_to_increment(target_px_raw, price_inc)
            cash_quote = float(self.get_cash() or 0.0)
            max_quote = float(p["max_inventory_quote"])  # cap buys to this exposure
            notional = min(cash_quote, max_quote)
            if target_px <= 0 or notional <= 0:
                self.log_message("Post-fill BUY skipped due to zero price/notional.", color="yellow")
            else:
                qty_raw = min(filled_qty, notional / target_px)
                qty = self._round_to_increment(qty_raw, qty_inc)
                dyn_min_qty = self._compute_min_qty(target_px)
                if dyn_min_qty is None:
                    self.log_message("Post-fill BUY skipped: min qty unavailable.", color="yellow")
                    return
                if qty < dyn_min_qty:
                    qty = self._round_to_increment(dyn_min_qty, qty_inc)
                if qty * target_px < 1.0 or qty < hard_min_qty:
                    self.log_message(
                        f"Post-fill BUY skipped: qty={qty:.6f}, price={target_px:.2f} fails min.",
                        color="yellow",
                    )
                else:
                    order_out = self.create_order(
                        self.base_asset,
                        qty,
                        Order.OrderSide.BUY,
                        order_type=Order.OrderType.LIMIT,
                        limit_price=float(target_px),
                        time_in_force="gtc",
                        quote=self.quote_asset_for_orders,
                    )
                    self.submit_order(order_out)
                    self.log_message(f"Post-fill BUY placed {qty:.6f} @ {target_px:.2f}", color="green")
                    try:
                        self.grid_logger.log_row({'event': 'NEW_ORDER', 'action': 'POST_FILL', 'side': 'BUY', 'qty': qty, 'price': float(target_px)})
                    except Exception:
                        pass
        # Save state after handling fills
        self._save_state()

    # -------------------- Parameters updated at runtime --------------------
    def on_parameters_updated(self, parameters: dict):
        # Allow re-enabling after a kill by toggling enabled=True
        if parameters.get("enabled", True) and self.vars.killed:
            self.vars.killed = False
            self.vars.flatten_executed = False
            self.log_message("Trading re-enabled via parameters.", color="green")

    # -------------------- Override tearsheet generation to prevent KDE errors --------------------
    def backtest_analysis(self, **kwargs):
        """Override to skip tearsheet generation that causes KDE errors with flat returns"""
        print(f"\nBacktest completed successfully!")
        
        # Get the actual portfolio value from the strategy's portfolio tracking
        final_value = self.get_portfolio_value()
        positions = self.get_positions()
        
        print(f"Final portfolio value: ${final_value:,.2f}")
        
        # Show positions if any exist
        if positions:
            print("Final positions:")
            for pos in positions:
                if pos.quantity != 0:
                    print(f"  {pos.asset.symbol}: {pos.quantity}")
        
        print("Skipping tearsheet generation to avoid statistical analysis errors.")
        return None


if __name__ == "__main__":
    # Users can change parameters by editing this dict or passing config when deploying
    params = CryptoGridMRStrategy.parameters

    if IS_BACKTESTING:
        # Backtesting path with Polygon for crypto data
        trading_fee = TradingFee(percent_fee=params.get("maker_fee_pct", 0.0015))

        # Use SPY as benchmark by convention; budget default is 100,000 unless user overrides
        result = CryptoGridMRStrategy.backtest(
            PolygonDataBacktesting,
            benchmark_asset=Asset("SPY", Asset.AssetType.STOCK),
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            parameters=params,
            budget=100000,
            start="2025-10-03",
            end="2025-10-06",
            show_plot=False,  # Disable plotting to avoid statistical errors
            show_tearsheet=False,  # Disable tearsheet to avoid KDE errors
            save_logfile=False,  # Disable log file saving
        )
    else:
        # Live trading path (broker is selected by environment variables; strategy is broker-agnostic)
        trader = Trader()
        strategy = CryptoGridMRStrategy(
            quote_asset=Asset("USD", Asset.AssetType.FOREX),  # Strategy quote context; actual order quotes use CRYPTO USD per initialize()
            parameters=params,
        )
        trader.add_strategy(strategy)
        strategies = trader.run_all()
