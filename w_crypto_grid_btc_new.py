#!/usr/bin/env python3
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset, Order, TradingFee
from lumibot.backtesting import PolygonDataBacktesting

import os
import json
from collections import deque
from statistics import stdev
from datetime import timedelta, datetime
from pathlib import Path

# Use the same approach as BTC_long.py for IS_BACKTESTING
IS_BACKTESTING = os.environ.get("IS_BACKTESTING", "False").lower() == "true"

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
                # Archive with yesterday's date
                archive = self.base_dir / f"grid_trading_log_{self._date.isoformat()}.csv"
                try:
                    fp.rename(archive)
                    # Keep only last 7 days of logs to prevent disk bloat
                    self._cleanup_old_logs()
                except Exception:
                    # If rename fails, continue to overwrite header next write
                    pass
            self._date = today

    def _cleanup_old_logs(self):
        """Remove log files older than 7 days"""
        try:
            cutoff_date = datetime.utcnow().date() - timedelta(days=7)
            for log_file in self.base_dir.glob("grid_trading_log_*.csv"):
                try:
                    # Extract date from filename: grid_trading_log_YYYY-MM-DD.csv
                    date_str = log_file.stem.split('_')[-1]  # Get the date part
                    file_date = datetime.fromisoformat(date_str).date()
                    if file_date < cutoff_date:
                        log_file.unlink()
                except (ValueError, IndexError):
                    # Skip files with unexpected naming format
                    continue
        except Exception:
            # Fail silently to not disrupt trading
            pass

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
        "ladders": 8,                  # reduced from 10 to concentrate capital on fewer, wider rungs
        # step_pct is now determined dynamically from ATR, see _compute_dynamic_step_pct
        "step_pct": 0.004,
        "envelope_k_atr": 2.5,         # bands = mid ± k * ATR (increased for wider bands)
        "rebuild_interval_sec": 60,    # how often to rebuild ladder (increased for trend filter)
        "max_spread_pct": 0.01,        # 1% spread safety
        "max_slippage_pct": 0.10,      # informational (orders are limit)

        # Trend filter settings
        "ema_window": 50,              # EMA period for trend detection
        "ema_slope_threshold": 0.002,  # 0.2% slope threshold for testing (reduced for more trading)
        "atr_window": 14,              # ATR calculation period (standard 14-period)
        "atr_factor": 0.5,             # multiplier for ATR-based step size
        "trail_drawdown_pct": 1.0,     # kill grid if equity drops this % from high

    # Fees (maker/taker) - updated per request
        "maker_fee_pct": 0.0015,       # 0.15%
        "taker_fee_pct": 0.0025,       # 0.25%

        # Risk limits
        "max_inventory_quote": 4000.0,   # cap net USD exposure (buy-side) - increased for better per-rung exposure
        "max_inventory_base": 0.4,       # cap net BTC inventory (sell-side) - increased proportionally
        "max_drawdown_pct": 10.0,        # kill if equity drawdown exceeds this
        "daily_loss_limit_quote": 300.0, # kill if daily loss exceeds this USD amount
        
        # Performance protection settings
        "trailing_drawdown_pct": 4.0,    # flatten positions if equity falls this % from peak
        "profit_target_pct": 8.0,        # take profits and pause when up this % from start
        "cooldown_period_hours": 6,      # hours to pause after hitting limits
        "losing_streak_limit": 3,        # max consecutive losing days before cooldown
        "min_performance_pct": -1.0,     # pause if rolling N-day performance below this
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
        # Tolerance to keep existing orders near targets (in whole price ticks).  Set higher to reduce churn.
        "target_tolerance_ticks": 20,  # increased to reduce order churn
        # Only rebuild ladder if mid price moves more than this percentage
        "rebuild_mid_threshold_pct": 0.0005,  # 0.05% threshold for testing (reduced for more frequent rebuilds)
    }

    def initialize(self):
        # Crypto trades 24/7; this keeps the bot running around the clock
        self.set_market("24/7")

        # Run frequent iterations by default (live); in backtests we align to minute bars below
        self.sleeptime = "1S"

        # Build asset objects once (store in self.vars as required by framework guidelines)
        p = self.get_parameters()
        self.base_asset = Asset(p["base_symbol"], asset_type=Asset.AssetType.CRYPTO)

        # IMPORTANT: Use FOREX for fiat/stable quote symbols so get_quote() returns valid bid/ask
        qs = p["quote_symbol"].upper()
        if qs in ("USD", "USDT", "USDC"):
            self.quote_asset_for_orders = Asset(p["quote_symbol"], asset_type=Asset.AssetType.FOREX)
        else:
            self.quote_asset_for_orders = Asset(p["quote_symbol"], asset_type=Asset.AssetType.CRYPTO)

        # Rolling price windows for EMA and ATR calculation
        # IMPORTANT: Keep these out of self.vars because Lumibot persists self.vars to JSON
        # and collections.deque is not JSON-serializable.
        self.price_window = deque(maxlen=max(p["ema_window"], p["atr_window"]) + 1)  # +1 for ATR calculation
        
        # For true EMA calculation
        self.ema_values = deque(maxlen=p["ema_window"])
        
        # For proper ATR calculation with true range
        self.atr_values = deque(maxlen=p["atr_window"])
        self.vars.last_high = None
        self.vars.last_low = None
        self.vars.last_close = None
        
        # Trailing equity tracking
        self.vars.tr_eq_high = self.get_portfolio_value() or 100000.0

        # State and persistence
        self.vars.last_rebuild_ts = None
        self.vars.killed = False
        self.vars.flatten_executed = False
        self.vars.ladder_prices = {"buys": [], "sells": []}  # last planned ladder levels
        self.vars.last_mid = None
        self.vars.last_quote_ts = None
        self.vars.needs_initial_reconcile = True
        # Stall debounce counter
        self.vars.stall_strikes = 0
        
        # Trend and volatility state
        self.vars.ema_current = None
        self.vars.ema_prev = None
        self.vars.atr_current = None
        self.vars.trend_active = False

        # Performance protection state - initialize with starting cash
        current_equity = self.cash  # Use starting cash as initial equity
        self.vars.equity_peak = current_equity  # Track highest equity reached
        self.vars.start_equity = current_equity  # Starting equity for profit target
        self.vars.paused_until = None            # Cooldown timestamp
        self.vars.consecutive_losing_days = 0    # Track losing streaks
        self.vars.last_day_pnl = 0.0            # Previous day's P&L

        # Risk tracking
        current_equity = self.get_portfolio_value() or 100000.0  # Use budget as fallback
        self.vars.daily_start_equity = current_equity
        self.vars.equity_highwater = current_equity
        self.vars.daily_date = self.get_datetime().date() if self.get_datetime() else None

        # Effective rebuild cadence (make backtests align to minute bars)
        self.vars.rebuild_interval_effective = int(p["rebuild_interval_sec"]) if int(p["rebuild_interval_sec"]) > 0 else 60

        if self.is_backtesting:
            # Align to minute bars in backtests so each iteration processes a completed bar
            self.sleeptime = "1M"  # 1 minute cadence for backtests
            self.vars.rebuild_interval_effective = 60  # rebuild once per bar
            self.log_message(
                f"Backtest mode: sleeptime=1M, rebuild_interval=60s.",
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
        if p["auto_clear_kill"] and p["enabled"] and self.vars.killed:
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
                self.vars.ladder_prices = data.get("ladder_prices", {"buys": [], "sells": []})
                # Only restore equity values if they seem reasonable
                current_equity = self.get_portfolio_value() or 100000.0
                saved_highwater = data.get("equity_highwater")
                saved_daily_start = data.get("daily_start_equity")
                
                # Reset equity tracking if saved values seem unreasonable
                if saved_highwater and saved_highwater > current_equity * 2:
                    self.vars.equity_highwater = current_equity
                else:
                    self.vars.equity_highwater = saved_highwater or current_equity
                    
                if saved_daily_start and saved_daily_start > current_equity * 2:
                    self.vars.daily_start_equity = current_equity
                else:
                    self.vars.daily_start_equity = saved_daily_start or current_equity
                # Restore trend state
                self.vars.ema_current = data.get("ema_current")
                self.vars.ema_prev = data.get("ema_prev")
                self.vars.atr_current = data.get("atr_current")
                self.vars.trend_active = bool(data.get("trend_active", False))
                
                # Restore performance protection state
                self.vars.equity_peak = data.get("equity_peak", current_equity)
                self.vars.start_equity = data.get("start_equity", current_equity)
                self.vars.paused_until = data.get("paused_until")
                self.vars.consecutive_losing_days = data.get("consecutive_losing_days", 0)
                self.vars.last_day_pnl = data.get("last_day_pnl", 0.0)
                self.log_message("State loaded from disk; will reconcile on first iteration.", color="yellow")
            except Exception as e:
                self.log_message(f"Failed to load state: {e}", color="red")

    def _save_state(self):
        data = {
            "killed": self.vars.killed,
            "last_mid": self.vars.last_mid,
            "ladder_prices": self.vars.ladder_prices,
            "equity_highwater": self.vars.equity_highwater,
            "daily_start_equity": self.vars.daily_start_equity,
            "ema_current": self.vars.ema_current,
            "ema_prev": self.vars.ema_prev,
            "atr_current": self.vars.atr_current,
            "trend_active": self.vars.trend_active,
            "equity_peak": self.vars.equity_peak,
            "start_equity": self.vars.start_equity,
            "paused_until": self.vars.paused_until,
            "consecutive_losing_days": self.vars.consecutive_losing_days,
            "last_day_pnl": self.vars.last_day_pnl,
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
            # Check losing streak before resetting daily values
            self._check_losing_streak()
            
            # Reset daily tracking
            self.vars.daily_date = now_dt.date()
            self.vars.daily_start_equity = self.get_portfolio_value() or 0.0
            self.log_message("Daily reset applied; new daily start equity set.", color="blue")

    def _risk_checks(self):
        p = self.get_parameters()
        if not p["enabled"]:
            return

        equity = self.get_portfolio_value() or 0.0
        
        # Update equity highwater
        if equity > (self.vars.equity_highwater or 0):
            self.vars.equity_highwater = equity
        
        # Incorporate trail stop (per user specification)
        equity = self.get_portfolio_value() or 0.0
        self.vars.tr_eq_high = max(self.vars.tr_eq_high, equity)
        
        if equity < self.vars.tr_eq_high * (1 - p["trail_drawdown_pct"]/100):
            print(f"DEBUG: Trailing drawdown kill: equity={equity:.2f} < {self.vars.tr_eq_high * (1 - p['trail_drawdown_pct']/100):.2f}")
            self._activate_kill_switch("Trailing drawdown hit")
            return

        # Update equity peak for trailing stop
        if equity > (self.vars.equity_peak or 0):
            self.vars.equity_peak = equity
            
        # Initialize daily_start_equity if not set
        if self.vars.daily_start_equity is None or self.vars.daily_start_equity == 0:
            self.vars.daily_start_equity = equity
            print(f"DEBUG: Initialized daily_start_equity to {equity}")
        
        # Initialize start_equity if not set
        if self.vars.start_equity is None or self.vars.start_equity == 0:
            self.vars.start_equity = equity
            
        # Calculate various drawdown and profit metrics
        if (self.vars.equity_highwater or 0) > 0:
            dd = (self.vars.equity_highwater - equity) / self.vars.equity_highwater
        else:
            dd = 0.0
            
        daily_loss = (self.vars.daily_start_equity or equity) - equity
        
        # Trailing drawdown from peak
        if (self.vars.equity_peak or 0) > 0:
            trailing_dd = (self.vars.equity_peak - equity) / self.vars.equity_peak
        else:
            trailing_dd = 0.0
            
        # Profit from start
        if (self.vars.start_equity or 0) > 0:
            total_profit_pct = (equity - self.vars.start_equity) / self.vars.start_equity
        else:
            total_profit_pct = 0.0

        print(f"DEBUG: Risk check - equity={equity:.2f}, peak={self.vars.equity_peak}, trailing_dd={trailing_dd*100:.2f}%, profit={total_profit_pct*100:.2f}%")

        self.log_message(
            f"Risk: equity=${equity:.0f}, trailing_dd={trailing_dd*100:.1f}%, profit={total_profit_pct*100:.1f}%",
            color="blue",
        )

        # Standard drawdown check
        if dd >= (p["max_drawdown_pct"] / 100.0):
            print(f"DEBUG: Drawdown kill triggered: {dd*100:.2f}% >= {p['max_drawdown_pct']}%")
            self._activate_kill_switch("Max drawdown breached")
            return
            
        # Daily loss check
        if daily_loss >= p["daily_loss_limit_quote"]:
            print(f"DEBUG: Daily loss kill triggered: {daily_loss:.2f} >= {p['daily_loss_limit_quote']}")
            self._activate_kill_switch("Daily loss limit breached")
            return
            
        # NEW: Trailing drawdown check
        if trailing_dd >= (p["trailing_drawdown_pct"] / 100.0):
            print(f"DEBUG: Trailing stop triggered: {trailing_dd*100:.2f}% >= {p['trailing_drawdown_pct']}%")
            self._activate_performance_pause("Trailing drawdown limit hit", p["cooldown_period_hours"])
            return
            
        # NEW: Profit target check
        if total_profit_pct >= (p["profit_target_pct"] / 100.0):
            print(f"DEBUG: Profit target hit: {total_profit_pct*100:.2f}% >= {p['profit_target_pct']}%")
            self._activate_performance_pause("Profit target reached", p["cooldown_period_hours"])
            return
        
        # Log risk snapshot
        try:
            self.grid_logger.log_row({
                'event': 'RISK',
                'equity': equity,
                'dd_pct': dd * 100.0,
                'daily_loss': daily_loss,
                'extra': f'trailing_dd={trailing_dd*100:.1f}%,profit={total_profit_pct*100:.1f}%'
            })
        except Exception:
            pass

    def _activate_performance_pause(self, reason: str, hours: float):
        """Pause trading for performance protection (less severe than kill switch)"""
        from datetime import datetime, timedelta
        
        # Cancel open orders and flatten positions
        self.cancel_open_orders()
        self._flatten_positions()
        
        # Set pause until timestamp
        pause_until = datetime.utcnow() + timedelta(hours=hours)
        self.vars.paused_until = pause_until.isoformat()
        
        self.log_message(f"PERFORMANCE PAUSE: {reason} - paused until {pause_until.strftime('%Y-%m-%d %H:%M')} UTC", color="yellow")
        
        try:
            self.grid_logger.log_row({
                'event': 'PERFORMANCE_PAUSE',
                'reason': reason,
                'extra': f'paused_until={self.vars.paused_until}'
            })
        except Exception:
            pass
        
        self._save_state()

    def _check_performance_cooldown(self) -> bool:
        """Check if we're still in a performance-based cooldown period"""
        if not self.vars.paused_until:
            return False
            
        from datetime import datetime
        try:
            pause_until = datetime.fromisoformat(self.vars.paused_until.replace('Z', '+00:00'))
            if datetime.utcnow() < pause_until.replace(tzinfo=None):
                return True
            else:
                # Cooldown period over
                self.vars.paused_until = None
                self.log_message("Performance cooldown period ended - resuming trading", color="green")
                self._save_state()
                return False
        except Exception:
            # If we can't parse the timestamp, assume cooldown is over
            self.vars.paused_until = None
            return False

    def _check_losing_streak(self):
        """Check for consecutive losing days and apply cooldown if needed"""
        p = self.get_parameters()
        equity = self.get_portfolio_value() or 0.0
        daily_start = self.vars.daily_start_equity or equity
        
        today_pnl = equity - daily_start
        
        # Check if this is a losing day
        if today_pnl < 0:
            if self.vars.last_day_pnl >= 0:  # Previous day was not losing
                self.vars.consecutive_losing_days = 1
            else:
                self.vars.consecutive_losing_days += 1
        else:
            # Reset streak on profitable day
            self.vars.consecutive_losing_days = 0
            
        self.vars.last_day_pnl = today_pnl
        
        # Check if we hit the losing streak limit
        if self.vars.consecutive_losing_days >= p["losing_streak_limit"]:
            self.log_message(f"Losing streak limit hit: {self.vars.consecutive_losing_days} consecutive losing days", color="red")
            self._activate_performance_pause("Consecutive losing days limit", p["cooldown_period_hours"])
            self.vars.consecutive_losing_days = 0  # Reset after triggering pause

    def _activate_kill_switch(self, reason: str):
        if self.vars.killed:
            return
        self.vars.killed = True
        self.log_message(f"KILL SWITCH ACTIVATED: {reason}", color="red")
        self.cancel_open_orders()
        if self.get_parameters()["flatten_on_kill"]:
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

    # -------------------- Utility: Market Data & Trend Analysis --------------------
    def _get_mid_and_spread(self):
        # Try to get a fresh quote with bid/ask; fallback to last price for backtesting
        now_dt = self.get_datetime()
        mid = None
        spread_pct = None
        reasons = []
        
        try:
            # Use get_last_price like the working BTC_long.py strategy
            last_price = self.get_last_price(self.base_asset)
            if last_price is not None and last_price > 0:
                mid = float(last_price)
                spread_pct = 0.001  # Default 0.1% spread for crypto
                self.vars.last_quote_ts = now_dt
                print(f"DEBUG: Got BTC price from get_last_price: ${mid}")
            else:
                print(f"DEBUG: get_last_price returned None or invalid price: {last_price}")
                reasons.append(f"get_last_price invalid: {last_price}")
        except Exception as e:
            print(f"DEBUG: get_last_price failed: {e}")
            reasons.append(f"get_last_price error: {e}")
            # Fallback to quote method
            try:
                if not self.is_backtesting:
                    q = self.get_quote(self.base_asset, quote=self.quote_asset_for_orders)
                else:
                    q = self.get_quote(self.base_asset)
                    
                if q is not None:
                    if q.bid is not None and q.ask is not None and q.bid > 0 and q.ask > 0 and q.ask >= q.bid:
                        mid = (q.bid + q.ask) / 2.0
                        if q.ask > 0 and mid and mid > 0:
                            spread_pct = (q.ask - q.bid) / mid
                        self.vars.last_quote_ts = now_dt
                        print(f"DEBUG: Got BTC price from quote: ${mid}")
            except Exception as e2:
                print(f"DEBUG: get_quote also failed: {e2}")
                reasons.append(f"get_quote error: {e2}")
        
        print(f"DEBUG: Final mid={mid}, spread_pct={spread_pct}")
        # Diagnostic CSV: if we failed to obtain a quote, record a NO_QUOTE event once per call
        if mid is None:
            try:
                self.grid_logger.log_row({
                    'event': 'NO_QUOTE',
                    'reason': '; '.join(reasons) if reasons else 'no price source available',
                })
            except Exception:
                pass
        return mid, spread_pct

    def _update_ema_and_trend(self, mid):
        """Update EMA and return True if market is trending"""
        params = self.get_parameters()
        self.ema_values.append(mid)
        
        if len(self.ema_values) < params["ema_window"]:
            return False  # not enough data yet
            
        # Calculate true exponential moving average
        if self.vars.ema_current is None:
            # Initialize EMA as simple average of first window
            self.vars.ema_current = sum(self.ema_values) / len(self.ema_values)
            self.vars.ema_prev = self.vars.ema_current
        else:
            # Update using exponential smoothing: α = 2/(N+1)
            alpha = 2.0 / (params["ema_window"] + 1)
            self.vars.ema_prev = self.vars.ema_current
            self.vars.ema_current = alpha * mid + (1 - alpha) * self.vars.ema_current
        
        # Calculate slope as percentage change
        if self.vars.ema_prev and self.vars.ema_current:
            slope = (self.vars.ema_current - self.vars.ema_prev) / self.vars.ema_prev
            return abs(slope) >= params["ema_slope_threshold"]
        
        return False

    def _update_atr(self, mid):
        """Track true range and return current ATR"""
        params = self.get_parameters()
        
        # Since we only have mid prices, approximate true range using price movements
        # True range = max(high-low, high-prev_close, low-prev_close)
        # Approximation: use current price movements and volatility
        if self.vars.last_mid is not None:
            # Simple true range approximation: absolute price change
            price_change = abs(mid - self.vars.last_mid)
            
            # Enhanced approximation: also consider intrabar volatility
            # Estimate high/low based on recent price movements
            if len(self.atr_values) > 0:
                recent_volatility = sum(self.atr_values) / len(self.atr_values)
                # Estimate true range as combination of price change and recent volatility
                estimated_high = max(mid, self.vars.last_mid) + recent_volatility * 0.1
                estimated_low = min(mid, self.vars.last_mid) - recent_volatility * 0.1
                true_range = max(
                    estimated_high - estimated_low,
                    abs(estimated_high - self.vars.last_mid) if self.vars.last_mid else 0,
                    abs(estimated_low - self.vars.last_mid) if self.vars.last_mid else 0
                )
            else:
                true_range = price_change
                
            self.atr_values.append(true_range)
            
        if len(self.atr_values) < params["atr_window"]:
            return None
            
        return sum(self.atr_values) / len(self.atr_values)
        
        # If trend just became active, cancel all orders and optionally flatten
        if self.vars.trend_active and not prev_trend_active:
            self.log_message(f"Trend activated (slope={slope:.6f}); canceling grid orders", color="red")
            self.cancel_open_orders()
            
        # If trend just became inactive, check volatility cap before resuming
        elif not self.vars.trend_active and prev_trend_active:
            if self.vars.atr_current and self.vars.last_mid:
                atr_ratio = self.vars.atr_current / self.vars.last_mid
                if atr_ratio < p["trend_volatility_cap"]:
                    self.log_message(f"Trend deactivated, low volatility (ATR/mid={atr_ratio:.4f}); resuming grid", color="green")
                else:
                    self.log_message(f"Trend deactivated but high volatility (ATR/mid={atr_ratio:.4f}); grid paused", color="yellow")
                    self.vars.trend_active = True  # Keep trend active due to high volatility

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
        """
        Return the minimum allowable step based on fees and edge.  Actual step size
        will be computed from volatility via _compute_dynamic_step_pct().
        """
        p = self.get_parameters()
        return 2.0 * p["maker_fee_pct"] + float(p["min_edge_pct"])

    def _compute_dynamic_step_pct(self, atr: float, mid: float) -> float:
        """
        Compute ATR-based step size. Step = (atr_factor * ATR) / mid_price
        Ensures the result exceeds the minimum step from fees.
        """
        p = self.get_parameters()
        min_step = self._enforced_step_pct()
        atr_factor = float(p["atr_factor"])
        if atr is None or mid <= 0:
            return min_step
        step_pct = (atr_factor * atr) / mid
        return max(min_step, step_pct)

    def _rebuild_ladder(self, mid, atr):
        p = self.get_parameters()
        if self.vars.killed or not p["enabled"]:
            self.log_message("Trading disabled or killed; skip ladder rebuild.", color="yellow")
            return

        if mid is None or mid <= 0:
            self.log_message("No valid mid price; skip ladder rebuild.", color="red")
            return

        # Check trend filter - skip ladder building if trend is active
        if self.vars.trend_active:
            self.log_message("Trend is active; skipping ladder rebuild", color="yellow")
            return

        # compute ATR-based step size
        step_pct = self._compute_dynamic_step_pct(atr, mid)
        if atr is None or atr <= 0:
            self.log_message("ATR not ready; skipping ladder rebuild this iteration.", color="yellow")
            return
        else:
            # ATR is in price units. Convert to price band around mid.
            k = p["envelope_k_atr"]
            # Ensure at least N step widths fit inside the envelope to host multiple rungs
            min_rungs = int(p["min_rungs_in_band"]) if int(p["min_rungs_in_band"]) > 0 else 1
            band_price = max(k * atr, step_pct * mid * min_rungs)
            lower = max(1e-9, mid - band_price)
            upper = mid + band_price

        # TEMPORARILY DISABLE threshold check to force rebuilds and test order placement
        mid_threshold = float(p["rebuild_mid_threshold_pct"])  # 0.05%
        print(f"THRESHOLD DEBUG: mid={mid:.2f}, last_mid={self.vars.last_mid}, threshold={mid_threshold:.6f}")
        print(f"THRESHOLD DEBUG: FORCING rebuild to test order placement (threshold disabled)")
        # Commenting out threshold check temporarily:
        # if self.vars.last_mid is not None:
        #     price_move = abs(mid - self.vars.last_mid) / self.vars.last_mid
        #     if price_move < mid_threshold:
        #         self.log_message(f"Mid move {price_move:.4%} below threshold; skipping ladder rebuild.", color="yellow")
        #         return

        # SELECTIVE ORDER MANAGEMENT

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
        self.log_message(f"Ladder rebuild: Mid={mid:.2f}, ATR={atr}, Step={step_pct:.4f}", color="cyan")
        self.log_message(f"Inventory: BTC={base_qty:.6f}, Cash=${cash_quote:.2f}", color="cyan")
        print(f"LADDER DEBUG: Mid={mid:.2f}, ATR={atr}, Cash=${cash_quote:.2f}, BTC={base_qty:.6f}")
        print(f"LADDER DEBUG: per_buy_notional=${per_buy_notional:.2f}, per_sell_qty={per_sell_qty:.6f}")
        print(f"LADDER DEBUG: buy_notional_cap=${buy_notional_cap:.2f}, sell_qty_cap={sell_qty_cap:.6f}, ladders={ladders}")
        print(f"LADDER DEBUG: max_quote=${max_quote:.2f}, max_base={max_base:.6f}")

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
        print(f"BUY DEBUG: target_buy_set={list(target_buy_set)[:5]}, kept_buy_prices={list(kept_buy_prices)[:5]}")
        print(f"BUY DEBUG: missing_buy_targets={missing_buy_targets[:5]}, per_buy_notional=${per_buy_notional:.2f}")
        if per_buy_notional >= 1.0:
            print(f"BUY DEBUG: per_buy_notional check passed, placing {len(missing_buy_targets)} buy orders")
            for limit_price in missing_buy_targets:
                if limit_price <= 0:
                    print(f"BUY DEBUG: Skipping limit_price={limit_price} (<=0)")
                    continue
                qty_raw = max(0.0, per_buy_notional / limit_price)
                qty = self._round_to_increment(qty_raw, qty_inc)
                print(f"BUY DEBUG: limit_price={limit_price:.2f}, qty_raw={qty_raw:.6f}, qty={qty:.6f}, dyn_min_qty={dyn_min_qty:.6f}")
                if qty < dyn_min_qty:
                    qty = self._round_to_increment(dyn_min_qty, qty_inc)
                    print(f"BUY DEBUG: Adjusted qty to min: {qty:.6f}")
                notional_value = qty * limit_price
                print(f"BUY DEBUG: Final check - notional_value=${notional_value:.2f}, qty={qty:.6f}, hard_min_qty={hard_min_qty:.6f}")
                if qty * limit_price < 1.0 or qty < hard_min_qty:
                    print(f"BUY DEBUG: REJECTED - notional=${notional_value:.2f} < 1.0 or qty={qty:.6f} < {hard_min_qty:.6f}")
                    self.log_message(
                        f"Skip NEW BUY @ {limit_price:.2f}: qty={qty:.6f} fails min ($1 or 0.0001).",
                        color="yellow",
                    )
                    continue
                print(f"BUY DEBUG: Creating order for {qty:.6f} @ ${limit_price:.2f}")
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
            print(f"BUY DEBUG: per_buy_notional check FAILED: ${per_buy_notional:.2f} < 1.0")
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
                'event': 'LADDER', 'mid': float(mid), 'atr': float(atr),
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
        self._save_state()

    # -------------------- Lifecycle: Core Loop --------------------
    def on_trading_iteration(self):
        print(f"ITERATION START - Portfolio: ${self.get_portfolio_value():.2f}")
        p = self.get_parameters()

        # If a stale kill flag was persisted, optionally clear it automatically when enabled=True
        if p["auto_clear_kill"] and p["enabled"] and self.vars.killed:
            self.vars.killed = False
            self.vars.flatten_executed = False
            self.log_message("Kill switch cleared automatically before iteration (auto_clear_kill=True).", color="green")
            print("DEBUG: Kill switch cleared")

        # Respect on/off switch
        if not p["enabled"]:
            print("DEBUG: Trading disabled")
            self.log_message("Trading is disabled via parameters; standing by.", color="yellow")
            return

        # Check for performance-based cooldown
        if self._check_performance_cooldown():
            print("DEBUG: In performance cooldown")
            self.log_message("In performance cooldown period; standing by.", color="yellow")
            return

        # If killed previously, keep idle (user can re-enable with parameters)
        if self.vars.killed:
            print("DEBUG: Kill switch active")
            if p["flatten_on_kill"] and not self.vars.flatten_executed:
                self._flatten_positions()
                self.vars.flatten_executed = True
            self.log_message("Kill switch is active; trading paused until re-enabled.", color="red")
            return

        print("DEBUG: Basic checks passed")

        # On first iteration after (re)start, reconcile by canceling stray orders
        if self.vars.needs_initial_reconcile:
            print("DEBUG: Initial reconcile")
            self.cancel_open_orders()
            self.vars.needs_initial_reconcile = False
            self.log_message("Initial reconcile complete: all stray orders canceled.", color="blue")

        # Daily reset and risk checks
        print("DEBUG: Running daily reset and risk checks")
        self._check_daily_reset()
        self._risk_checks()
        if self.vars.killed:
            print("DEBUG: Killed by risk checks")
            return
        # Check for performance cooldown again after risk checks
        if self._check_performance_cooldown():
            print("DEBUG: Performance pause triggered by risk checks")
            return

        # Market data: get mid & spread
        print("DEBUG: Getting market data")
        mid, spread_pct = self._get_mid_and_spread()
        print(f"DEBUG: Market data: Mid=${mid}, Spread={spread_pct}")
        
        if mid is None:
            print("DEBUG: No price data available")
            self.log_message("No price available this iteration; waiting for data.", color="yellow")
            return

        print("DEBUG: About to update trend indicators")

        # Update trend filter
        trending = self._update_ema_and_trend(mid)
        if trending:
            print(f"DEBUG: Market trending detected - EMA slope exceeded {p['ema_slope_threshold']}")
            self.log_message("Market trending; pausing grid.", color="yellow")
            return
        else:
            print(f"DEBUG: Market NOT trending - safe to trade")
        
        # Update ATR and compute step
        atr = self._update_atr(mid)
        if atr is not None:
            step_pct = max(self._enforced_step_pct(), (atr * p["atr_factor"]) / mid)
        else:
            print("ATR not ready - skipping this iteration")
            self.log_message("ATR not ready (need more samples); skipping this iteration.", color="yellow")
            return
            
        # Store ATR for ladder rebuild
        self.vars.atr_current = atr
        
        # Update last_mid for next ATR calculation
        self.vars.last_mid = mid

        # Apply stall kill-switch only in live trading
        now_dt = self.get_datetime()
        print(f"Checking stall kill-switch: is_backtesting={self.is_backtesting}")
        if (not self.is_backtesting) and self.vars.last_quote_ts and now_dt:
            stall = (now_dt - self.vars.last_quote_ts).total_seconds()
            if stall > p["kill_switch_on_ws_stall_sec"]:
                # Increment stall strikes and only kill after N consecutive stalls
                self.vars.stall_strikes = int(getattr(self.vars, 'stall_strikes', 0)) + 1
                strikes_needed = int(p["stall_kill_strikes"])
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
        print(f"Using ATR={atr} for grid spacing")

        # Rebuild ladder only every effective interval (60s in backtests to match minute bars)
        ts = self.get_timestamp()
        interval_sec = self.vars.rebuild_interval_effective
        if self.vars.last_rebuild_ts is None or (ts - self.vars.last_rebuild_ts) >= interval_sec:
            print(f"REBUILDING LADDER: Mid=${mid:.2f}, ATR={atr}")
            self.log_message(f"REBUILDING LADDER: Mid=${mid:.2f}, ATR={atr}", color="yellow")
            self._rebuild_ladder(mid, atr)
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
        
        # Check for trend breakout beyond current envelope
        if self.vars.trend_active and self.vars.atr_current and self.vars.last_mid:
            p = self.get_parameters()
            k = p["envelope_k_atr"]
            band_price = k * self.vars.atr_current
            lower = self.vars.last_mid - band_price
            upper = self.vars.last_mid + band_price
            
            fill_price = float(price) if price is not None else None
            if fill_price and (fill_price < lower or fill_price > upper):
                self.log_message(f"Trend breakout detected: fill at {fill_price:.2f} beyond envelope [{lower:.2f}, {upper:.2f}]", color="red")
                self._activate_kill_switch("Trend breakout beyond grid")
                return
        
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
        if self.vars.killed or not p["enabled"]:
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
        if parameters["enabled"] and self.vars.killed:
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
        trading_fee = TradingFee(percent_fee=params["maker_fee_pct"])

        # Get dates from environment variables
        start_date = os.getenv("BACKTESTING_START")
        end_date = os.getenv("BACKTESTING_END")

        # Use SPY as benchmark by convention; budget default is 100,000 unless user overrides
        result = CryptoGridMRStrategy.backtest(
            PolygonDataBacktesting,
            benchmark_asset=Asset("SPY", Asset.AssetType.STOCK),
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            parameters=params,
            budget=100000,
            start=start_date,
            end=end_date,
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
