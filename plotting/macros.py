TEMPLATE = "simple_white"

IPA = "IPA-like (off-device)"
COOKIEMONSTER_BASE = "ARA-like (on-device)"
COOKIEMONSTER = "Cookie Monster"
COOKIEMONSTER_BIAS = "Cookie Monster w/ bias detection"

KNOB1_AXIS = "fraction of users per query"
KNOB2_AXIS = "user impressions per day"

IMPRESSIONS_COUNT_X = "# of impressions"
EPOCHS_COUNT_X = "# of epochs"

BUDGET_CONSUMPTION_X = "# of queries submitted"
BUDGET_CONSUMPTION_Y = "consumed budget"
BUDGET_CONSUMPTION_Y_MAX = "max budget"
BUDGET_CONSUMPTION_Y_MAX_AVG = "avg. max consumed budget"
BUDGET_CONSUMPTION_Y_AVG_MAX = "max avg. consumed budget"
BUDGET_CONSUMPTION_Y_AVG_LOG = "avg. budget (log)"
BUDGET_CONSUMPTION_Y_AVG = "avg. budget"
BUDGET_TIME_Y_AVG = "report creation time (ms)"

SOURCE_TIME_Y_AVG = "avg. source matching time (ms)"

RMSRE_CDF_X = "% of queries"

BUDGET_CDF_X = "% of devices (for all queriers)"
BUDGET_CDF_Y = "budget consumption"

NUM_DAYS_PER_EPOCH_X = "# of days per epoch"
INITIAL_BUDGET_X = "initial budget"
RMSRE_Y = "RMSRE query error"
RMSRE_Y_LOG = "RMSRE query error (log)"

CUSTOM_ORDER_BASELINES = [IPA, COOKIEMONSTER_BASE, COOKIEMONSTER]
CUSTOM_ORDER_RATES = ["0.001", "0.01", "0.1", "1.0"]

AXIS_FONT_SIZE = 20

color_discrete_map = {
    COOKIEMONSTER: "blue",
    IPA: "red",
    COOKIEMONSTER_BASE: "green",
    COOKIEMONSTER_BIAS: "purple",
    
}

symbol_map = {
    COOKIEMONSTER: "circle",
    IPA: "x",
    COOKIEMONSTER_BASE: "diamond",
    COOKIEMONSTER_BIAS: "circle",
    
}


lines_map = {
    COOKIEMONSTER: "solid",
    IPA: "dash",
    COOKIEMONSTER_BASE: "dot",
    COOKIEMONSTER_BIAS: "solid",
    
}

DEFAULT_BASELINES_ORDER = [COOKIEMONSTER, COOKIEMONSTER_BASE, IPA]
BIAS_BASELINES_ORDER = [COOKIEMONSTER, COOKIEMONSTER_BIAS, COOKIEMONSTER_BASE, IPA]

csv_mapping = {
    IPA: "ipa",
    COOKIEMONSTER: "cookiemonster",
    COOKIEMONSTER_BASE: "cookiemonster_base",
    COOKIEMONSTER_BIAS: "cookiemonster_bias",
}

# ['', '/', '\\', 'x', '-', '|', '+', '.']
pattern_shape_map = {
    COOKIEMONSTER: "/",
    IPA: "x",
    COOKIEMONSTER_BASE: "\\",
}


def ensure_ordering(df, attr, type="str"):
    df = df.sort_values(by=[attr])
    df = df.astype({attr: type})
    return df
