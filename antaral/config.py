"""Shared constants for Antaral.

All potassium values are serum potassium in mEq/L (equivalent to mmol/L).
Time runs in whole hours; day 0 of every simulation is a Monday.
"""

# --- Adverse-event definition -------------------------------------------------
K_THRESHOLD = 6.0   # moderate-to-severe hyperkalaemia: the event Antaral predicts
K_SEVERE = 6.5      # severe hyperkalaemia: reported separately in evaluations
K_CRITICAL = 7.0    # critical: the range where fatal arrhythmias become much more likely

# --- Dialysis unit ------------------------------------------------------------
BATH_K = 2.0                 # dialysate potassium bath, mEq/L
SESSION_HOURS = 4            # standard session length
SHIFT_STARTS = (7, 11, 16)   # three shifts a day; the last one ends at 20:00
DECISION_HOUR = 21           # the unit plans tomorrow's list at 9 PM
CLOSED_WEEKDAY = 6           # Sunday: unit closed
MAX_WINDOW_H = 96            # longest look-ahead used by the risk model

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Weekly patterns. Twice-weekly is very common in India for cost and capacity reasons.
PATTERNS = {
    "Mon-Thu": (0, 3),
    "Tue-Fri": (1, 4),
    "Wed-Sat": (2, 5),
    "Mon-Wed-Fri": (0, 2, 4),
    "Tue-Thu-Sat": (1, 3, 5),
}
MIN_GAP_DAYS = {2: 2, 3: 1}   # shortest allowed gap between sessions
MAX_GAP_DAYS = {2: 4, 3: 3}   # longest allowed gap (the usual long weekend gap)

# --- Potassium-rich foods common in Indian households (approximate mEq per serving)
# 1 mEq K = 39.1 mg. A tender coconut holds roughly 300-500 ml at ~250 mg K / 100 ml.
MG_PER_MEQ_K = 39.1
K_ITEMS = {
    "Tender coconut water": 22.0,
    "Banana": 10.5,
    "Orange / mosambi juice": 11.0,
    "Dry fruits (handful)": 13.0,
    "Aloo sabzi": 9.0,
    "Tomato rasam / sambar": 7.0,
}
K_ITEM_PROBS = [0.24, 0.22, 0.14, 0.12, 0.16, 0.12]

# Fraction of the day's food potassium eaten at each hour (meals + evening tea).
MEAL_FRACTIONS = {8: 0.27, 13: 0.36, 17: 0.07, 20: 0.30}

# Wearable recording times
ECG_HOURS = (7, 21)       # smartwatch single-lead ECG spot checks
WEIGHT_HOUR = 6           # smart-scale morning weight
VITALS_SYNC_HOUR = 20     # daily wearable summary (steps, resting HR, HRV, sleep, BP)
