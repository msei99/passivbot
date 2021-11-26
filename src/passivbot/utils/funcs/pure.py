from __future__ import annotations

import datetime
import pprint
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
from dateutil import parser

from passivbot.datastructures import Fill
from passivbot.datastructures import Order
from passivbot.utils.funcs.njit import qty_to_cost
from passivbot.utils.funcs.njit import round_dynamic


def format_float(num) -> str:
    formatted: str = np.format_float_positional(num, trim="-")
    return formatted


def compress_float(n: float, d: int) -> str:
    if n / 10 ** d >= 1:
        n = round(n)
    else:
        n = round_dynamic(n, d)
    nstr = format_float(n)
    if nstr.startswith("0."):
        nstr = nstr[1:]
    elif nstr.startswith("-0."):
        nstr = "-" + nstr[2:]
    elif nstr.endswith(".0"):
        nstr = nstr[:-2]
    return nstr


def calc_spans(min_span: int, max_span: int, n_spans: int) -> np.ndarray:
    return np.array(
        [min_span * ((max_span / min_span) ** (1 / (n_spans - 1))) ** i for i in range(0, n_spans)]
    )
    return np.array([min_span, (min_span * max_span) ** 0.5, max_span])


def numpyize(x):
    if isinstance(x, (list, tuple)):
        return np.array([numpyize(e) for e in x])
    elif isinstance(x, dict):
        numpyd = {}
        for k, v in x.items():
            numpyd[k] = numpyize(v)
        return numpyd
    else:
        return x


def denumpyize(x: Any) -> Any:
    if isinstance(x, (np.float64, np.float32, np.float16)):
        return float(x)
    elif isinstance(x, (np.int64, np.int32, np.int16, np.int8)):
        return int(x)
    elif isinstance(x, np.ndarray):
        return [denumpyize(e) for e in x]
    elif isinstance(x, np.bool_):
        return bool(x)
    elif isinstance(x, (dict, OrderedDict)):
        denumpyd = {}
        for k, v in x.items():
            denumpyd[k] = denumpyize(v)
        return denumpyd
    elif isinstance(x, list):
        return [denumpyize(z) for z in x]
    elif isinstance(x, tuple):
        return tuple(denumpyize(z) for z in x)
    else:
        return x


def denanify(x, nan=0.0, posinf=0.0, neginf=0.0):
    try:
        assert not isinstance(x, str)
        _ = float(x)
        return np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    except Exception:
        if isinstance(x, list):
            return [denanify(e) for e in x]
        elif isinstance(x, tuple):
            return tuple(denanify(e) for e in x)
        elif isinstance(x, np.ndarray):
            return np.array([denanify(e) for e in x], dtype=x.dtype)
        elif isinstance(x, dict):
            denanified = {}
            for k, v in x.items():
                denanified[k] = denanify(v)
            return denanified
        else:
            return x


def ts_to_date(timestamp: float, cast_to_str: bool = True) -> str | datetime.datetime:
    if timestamp > 253402297199:
        value = datetime.datetime.fromtimestamp(timestamp / 1000)
    else:
        value = datetime.datetime.fromtimestamp(timestamp)
    if cast_to_str:
        return str(value).replace(" ", "T")
    return value


def date_to_ts(d):
    return int(parser.parse(d).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def get_utc_now_timestamp() -> int:
    """
    Creates a millisecond based timestamp of UTC now.
    :return: Millisecond based timestamp of UTC now.
    """
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)


def config_pretty_str(config: dict[str, Any]):
    pretty_str = pprint.pformat(config)
    for r in [("'", '"'), ("True", "true"), ("False", "false")]:
        pretty_str = pretty_str.replace(*r)
    return pretty_str


def candidate_to_live_config(candidate: dict[str, Any]) -> Any:
    packed = pack_config(candidate)
    live_config = get_template_live_config()
    sides = ["long", "short"]
    for side in sides:
        for k in live_config[side]:
            if k in packed[side]:
                live_config[side][k] = packed[side][k]
    for k in live_config:
        if k not in sides and k in packed:
            live_config[k] = packed[k]

    result_dict = candidate["result"] if "result" in candidate else candidate
    if packed["long"]["enabled"] and not packed["short"]["enabled"]:
        side_type = "longonly"
    elif packed["short"]["enabled"] and not packed["long"]["enabled"]:
        side_type = "shortonly"
    else:
        side_type = "long&short"
    name = f"{side_type}_"
    name += f"{result_dict['exchange'].lower()}_" if "exchange" in result_dict else "unknown_"
    name += f"{result_dict['symbol'].lower()}" if "symbol" in result_dict else "unknown"
    if "n_days" in result_dict:
        n_days = result_dict["n_days"]
    elif "start_date" in result_dict:
        n_days = (date_to_ts(result_dict["end_date"]) - date_to_ts(result_dict["start_date"])) / (
            1000 * 60 * 60 * 24
        )
    else:
        n_days = 0
    name += f"_{n_days:.0f}days"
    if "average_daily_gain" in result_dict:
        name += f"_adg{(result_dict['average_daily_gain']) * 100:.3f}%"
    elif "daily_gain" in result_dict:
        name += f"_adg{(result_dict['daily_gain'] - 1) * 100:.3f}%"
    if "closest_bkr" in result_dict:
        name += f"_bkr{(result_dict['closest_bkr']) * 100:.2f}%"
    if "eqbal_ratio_min" in result_dict:
        name += f"_eqbal{(result_dict['eqbal_ratio_min']) * 100:.2f}%"
    live_config["config_name"] = name
    return denumpyize(live_config)


def unpack_config(d):
    new = {}
    for k, v in flatten_dict(d, sep="£").items():
        try:
            assert not isinstance(v, str)
            for _ in v:
                break
            for i in range(len(v)):
                new[f"{k}${str(i).zfill(2)}"] = v[i]
        except Exception:
            new[k] = v
    if new == d:
        return new
    return unpack_config(new)


def pack_config(d):
    while any("$" in k for k in d):
        new: dict[str, Any] = {}
        for k, v in denumpyize(d).items():
            if "$" in k:
                ks = k.split("$")
                k0 = "$".join(ks[:-1])
                if k0 in new:
                    new[k0].append(v)
                else:
                    new[k0] = [v]
            else:
                new[k] = v
        d = new
    new = {}
    for k, v in d.items():
        if isinstance(v, list):
            new[k] = np.array(v)
        else:
            new[k] = v
    d = new

    new = {}
    for k, v in d.items():
        if "£" in k:
            k0, k1 = k.split("£")
            if k0 in new:
                new[k0][k1] = v
            else:
                new[k0] = {k1: v}
        else:
            new[k] = v
    return new


def flatten_dict(d: dict[str, Any], parent_key="", sep="_") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def sort_dict_keys(d: Any) -> Any:
    if isinstance(d, list):
        return [sort_dict_keys(e) for e in d]
    if not isinstance(d, dict):
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def filter_orders(
    actual_orders: list[Order],
    ideal_orders: list[Order],
    keys: tuple[str, ...] | list[str] = ("symbol", "side", "qty", "price"),
) -> tuple[list[Order], list[Order]]:
    # returns (orders_to_delete, orders_to_create)

    if not actual_orders:
        return [], ideal_orders
    if not ideal_orders:
        return actual_orders, []
    actual_orders = actual_orders.copy()
    orders_to_create = []
    ideal_orders_cropped = [{k: getattr(o, k) for k in keys} for o in ideal_orders]
    actual_orders_cropped = [{k: getattr(o, k) for k in keys} for o in actual_orders]
    for ioc, io in zip(ideal_orders_cropped, ideal_orders):
        matches = [(aoc, ao) for aoc, ao in zip(actual_orders_cropped, actual_orders) if aoc == ioc]
        if matches:
            actual_orders.remove(matches[0][1])
            actual_orders_cropped.remove(matches[0][0])
        else:
            orders_to_create.append(io)
    return actual_orders, orders_to_create


def get_dummy_settings(config: dict[str, Any]) -> dict[str, Any]:
    dummy_settings = get_template_live_config()
    dummy_settings.update(
        {
            "user": config["user"],
            "exchange": config["exchange"],
            "symbol": config["symbol"],
            "config_name": "",
            "logging_level": 0,
        }
    )
    return {**config, **dummy_settings}


def flatten(lst: list[Any]) -> list[Any]:
    return [y for x in lst for y in x]


def get_template_live_config():
    return {
        "config_name": "template",
        "logging_level": 0,
        "long": {
            "enabled": True,
            "grid_span": 0.16,
            "wallet_exposure_limit": 1.6,
            "max_n_entry_orders": 10,
            "initial_qty_pct": 0.01,
            "eprice_pprice_diff": 0.0025,
            "secondary_allocation": 0.5,
            "secondary_pprice_diff": 0.35,
            "eprice_exp_base": 1.618034,
            "min_markup": 0.0045,
            "markup_range": 0.0075,
            "n_close_orders": 7,
        },
        "short": {
            "enabled": False,
            "grid_span": 0.16,
            "wallet_exposure_limit": 1.6,
            "max_n_entry_orders": 10,
            "initial_qty_pct": 0.01,
            "eprice_pprice_diff": 0.0025,
            "secondary_allocation": 0.5,
            "secondary_pprice_diff": 0.35,
            "eprice_exp_base": 1.618034,
            "min_markup": 0.0045,
            "markup_range": 0.0075,
            "n_close_orders": 7,
        },
    }


def analyze_fills(
    fills: list[Fill],
    stats: list[Any],
    inverse: bool,
    c_mult: float,
    exchange: str | None = None,
    symbol: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sdf = pd.DataFrame(
        stats,
        columns=[
            "timestamp",
            "balance",
            "equity",
            "bkr_price",
            "long_psize",
            "long_pprice",
            "short_psize",
            "short_pprice",
            "price",
            "closest_bkr",
        ],
    )
    fdf = pd.DataFrame(
        fills,
        columns=[
            "trade_id",
            "timestamp",
            "pnl",
            "fee_paid",
            "balance",
            "equity",
            "qty",
            "price",
            "psize",
            "pprice",
            "type",
        ],
    )
    fdf.loc[:, "wallet_exposure"] = [
        qty_to_cost(x.psize, x.pprice, inverse, c_mult) / x.balance if x.balance > 0.0 else 0.0
        for x in fdf.itertuples()
    ]
    sdf.loc[:, "long_wallet_exposure"] = [
        qty_to_cost(x.long_psize, x.long_pprice, inverse, c_mult) / x.balance
        if x.balance > 0.0
        else 0.0
        for x in sdf.itertuples()
    ]
    sdf.loc[:, "short_wallet_exposure"] = [
        qty_to_cost(x.short_psize, x.short_pprice, inverse, c_mult) / x.balance
        if x.balance > 0.0
        else 0.0
        for x in sdf.itertuples()
    ]
    gain = sdf.balance.iloc[-1] / sdf.balance.iloc[0]
    n_days = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[0]) / (1000 * 60 * 60 * 24)
    adg = gain ** (1 / n_days) - 1
    gain -= 1
    fills_per_day = len(fills) / n_days
    long_pos_changes = sdf[sdf.long_psize != sdf.long_psize.shift()]
    long_pos_changes_ms_diff = np.diff(
        [sdf.timestamp.iloc[0]] + list(long_pos_changes.timestamp) + [sdf.timestamp.iloc[-1]]
    )
    hrs_stuck_max_long = long_pos_changes_ms_diff.max() / (1000 * 60 * 60)
    hrs_stuck_avg_long = long_pos_changes_ms_diff.mean() / (1000 * 60 * 60)
    lpprices = sdf[sdf.long_pprice != 0.0]
    pa_distance_long = (lpprices.long_pprice - lpprices.price).abs() / lpprices.price
    analysis = {
        "exchange": exchange or "unknown",
        "symbol": symbol or "unknown",
        "starting_balance": sdf.balance.iloc[0],
        "pa_distance_mean_long": pa_distance_long.mean(),
        "pa_distance_median_long": pa_distance_long.median(),
        "pa_distance_max_long": pa_distance_long.max(),
        "average_daily_gain": adg,
        "adjusted_daily_gain": np.tanh(20 * adg) / 20,
        "gain": gain,
        "n_days": n_days,
        "n_fills": len(fills),
        "n_entries": len(fdf[fdf.type.str.contains("entry")]),
        "n_closes": len(fdf[fdf.type.str.contains("close")]),
        "n_ientries": len(fdf[fdf.type.str.contains("ientry")]),
        "n_rentries": len(fdf[fdf.type.str.contains("rentry")]),
        "avg_fills_per_day": fills_per_day,
        "hrs_stuck_max_long": hrs_stuck_max_long,
        "hrs_stuck_avg_long": hrs_stuck_avg_long,
        "hrs_stuck_max": hrs_stuck_max_long,
        "hrs_stuck_avg": hrs_stuck_avg_long,
        "loss_sum": fdf[fdf.pnl < 0.0].pnl.sum(),
        "profit_sum": fdf[fdf.pnl > 0.0].pnl.sum(),
        "pnl_sum": (pnl_sum := fdf.pnl.sum()),
        "fee_sum": (fee_sum := fdf.fee_paid.sum()),
        "net_pnl_plus_fees": pnl_sum + fee_sum,
        "final_equity": sdf.equity.iloc[-1],
        "final_balance": sdf.balance.iloc[-1],
        "closest_bkr": sdf.closest_bkr.min(),
        "eqbal_ratio_min": (eqbal_ratios := sdf.equity / sdf.balance).min(),
        "eqbal_ratio_mean": eqbal_ratios.mean(),
        "eqbal_ratio_median": eqbal_ratios.median(),
        "biggest_psize": fdf.psize.abs().max(),
    }
    return fdf, sdf, sort_dict_keys(analysis)


def calc_pprice_from_fills(coin_balance, fills: list[Fill], n_fills_limit=100):
    # assumes fills are sorted old to new
    if coin_balance == 0.0 or len(fills) == 0:
        return 0.0
    relevant_fills = []
    qty_sum = 0.0
    for fill in fills[::-1][:n_fills_limit]:
        abs_qty = fill.qty
        if fill.side == "buy":
            adjusted_qty = min(abs_qty, coin_balance - qty_sum)
            qty_sum += adjusted_qty
            rfill = fill.copy()
            rfill.qty = adjusted_qty
            relevant_fills.append(rfill)
            if qty_sum >= coin_balance * 0.999:
                break
        else:
            qty_sum -= abs_qty
            relevant_fills.append(fill)
    psize, pprice = 0.0, 0.0
    for fill in relevant_fills[::-1]:
        abs_qty = abs(fill.qty)
        if fill.side == "buy":
            new_psize = psize + abs_qty
            pprice = pprice * (psize / new_psize) + fill.price * (abs_qty / new_psize)
            psize = new_psize
        else:
            psize -= abs_qty
    return pprice


def get_position_fills(
    long_psize: float, short_psize: float, fills: list[Fill]
) -> tuple[list[Fill], list[Fill]]:
    """
    assumes fills are sorted old to new
    returns fills since and including initial entry
    """
    long_psize *= 0.999
    short_psize *= 0.999
    long_qty_sum = 0.0
    short_qty_sum = 0.0
    long_done, short_done = long_psize == 0.0, short_psize == 0.0
    if long_done and short_done:
        return [], []
    long_pfills, short_pfills = [], []
    for x in fills[::-1]:
        if x.position_side == "long":
            if not long_done:
                long_qty_sum += x.qty * (1.0 if x.side == "buy" else -1.0)
                long_pfills.append(x)
                long_done = long_qty_sum >= long_psize
        elif x.position_side == "short":
            if not short_done:
                short_qty_sum += x.qty * (1.0 if x.side == "sell" else -1.0)
                short_pfills.append(x)
                short_done = short_qty_sum >= short_psize
    return long_pfills[::-1], short_pfills[::-1]


def calc_long_pprice(long_psize, long_pfills: list[Fill]) -> float:
    """
    assumes long pfills are sorted old to new
    """
    psize, pprice = 0.0, 0.0
    for fill in long_pfills:
        abs_qty = abs(fill.qty)
        if fill.side == "buy":
            new_psize = psize + abs_qty
            pprice = pprice * (psize / new_psize) + fill.price * (abs_qty / new_psize)
            psize = new_psize
        else:
            psize = max(0.0, psize - abs_qty)
    return pprice


def nullify(x):
    if isinstance(x, (list, tuple)):
        return [nullify(x1) for x1 in x]
    elif isinstance(x, np.ndarray):
        return numpyize([nullify(x1) for x1 in x])
    elif isinstance(x, dict):
        return {k: nullify(x[k]) for k in x}
    elif isinstance(x, (bool, np.bool_)):
        return x
    else:
        return 0.0


def spotify_config(config: dict[str, Any], nullify_short: bool = True) -> dict[str, Any]:
    spotified = config.copy()

    spotified["spot"] = True
    if "market_type" not in spotified:
        spotified["market_type"] = "spot"
    elif "spot" not in spotified["market_type"]:
        spotified["market_type"] += "_spot"
    spotified["do_long"] = spotified["long"]["enabled"] = config["long"]["enabled"]
    spotified["do_short"] = spotified["short"]["enabled"] = False
    spotified["long"]["wallet_exposure_limit"] = min(
        1.0, spotified["long"]["wallet_exposure_limit"]
    )
    if nullify_short:
        spotified["short"] = nullify(spotified["short"])
    return spotified


def tuplify(xs):
    if isinstance(xs, list):
        return tuple(tuplify(x) for x in xs)
    elif isinstance(xs, dict):
        return tuple({k: tuplify(v) for k, v in xs.items()}.items())
    return xs


def round_values(xs, n: int):
    if isinstance(xs, (float, np.float64)):
        return round_dynamic(xs, n)
    if isinstance(xs, dict):
        return {k: round_values(xs[k], n) for k in xs}
    if isinstance(xs, list):
        return [round_values(x, n) for x in xs]
    if isinstance(xs, np.ndarray):
        return numpyize([round_values(x, n) for x in xs])
    if isinstance(xs, tuple):
        return tuple(round_values(x, n) for x in xs)
    if isinstance(xs, OrderedDict):
        return OrderedDict([(k, round_values(xs[k], n)) for k in xs])
    return xs


def floatify(xs):
    try:
        return float(xs)
    except (ValueError, TypeError):
        if isinstance(xs, list):
            return [floatify(x) for x in xs]
        if isinstance(xs, dict):
            return {k: floatify(v) for k, v in xs.items()}
        if isinstance(xs, tuple):
            return tuple(floatify(x) for x in xs)
    return xs


def get_daily_from_income(
    income: list[dict[str, Any]],
    balance: float,
    start_time: int | None = None,
    end_time: int | None = None,
):
    if start_time is None:
        start_time = income[0]["timestamp"]
    if end_time is None:
        end_time = income[-1]["timestamp"]
    idf = pd.DataFrame(income)
    idf.loc[:, "datetime"] = idf.timestamp.apply(ts_to_date)
    idf.index = idf.timestamp
    ms_per_day = 1000 * 60 * 60 * 24
    days = idf.timestamp // ms_per_day * ms_per_day
    groups = idf.groupby(days)
    daily_income = (
        groups.income.sum()
        .reindex(
            np.arange(
                start_time // ms_per_day * ms_per_day,
                end_time // ms_per_day * ms_per_day + ms_per_day,
                ms_per_day,
            )
        )
        .fillna(0.0)
    )
    cumulative = daily_income.cumsum()
    starting_balance = balance - cumulative.iloc[-1]
    plus_balance = cumulative + starting_balance
    daily_pct = daily_income / plus_balance
    bdf = pd.DataFrame(
        {
            "abs_income": daily_income.values,
            "gain": daily_pct.values,
            "cumulative": plus_balance.values,
        },
        index=[ts_to_date(x) for x in daily_income.index],
    )
    return idf, bdf
