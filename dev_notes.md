## Vision Anchor for SHRM development
SHRM is a Alpha Research Tool

This projects purpose is:
 - To get valid data for later projects
 - Create a real, testable edge
 - Expand my project portfolio

This model should output 2 values:
 - The expected return of the fair market price
 - The model's confidence of that prediction


This project does not accomplish:
 - Full risk Assessment
 - Polished product-ready GUI
 - Tracking holdings
 - Create Stop Loss and Take Profit

## Inputs
This model will input 4 values:
 - Volatility Z score
 - Log Volume
 - Drawdown State
 - Price Slope

## Outputs
Future Return (24) (μ)
 - Defined by: The models predicted return of (24) hours
 - df["feture_ret_24"] = df["log_ret"].rolling(24).sum().shift(-24)
Model Confidence (C)
 - Defined by: A measure of how certain the model is on its μ prediction
 - Higher Uncertainty -> Lower C
 - Lower Uncertainty -> Higher C
 - Mathmatical Definition:
 - rt,H = actual cumulative log return over next H hours
 - μt = predicted expected return
 - σt = predicted standard deviation (uncertainty)
 - ====================================
 - The model outputs:
 - μt, log(sqr(σt))
 - Uncertainty:
 - σt = exp(0.5*log(sqr(σt)))
 - Logg function(Gaussian negative log-likelihood):
 - Lt = sqr(rt,h - μt) / sqr(σt) + log(sqr(σt))
 - Confidence Definition:
 - Ct = 1/1 + (σt/s)
 - s Definiton:
 - s = std(rt,H) from training data
 - Range:
 - Ct​∈(0,1]
 - If: σt -> 0 then Ct -> 1
 - If: σt is higher, then Ct -> 0
 - Higher C = more confident position
 - =================================
 - Coding definition:
 - mu, log_var = model(x)
 - Compute:
 - sigma = torch.exp(0.5 * log_var)
 - confidence = 1 / (1 + sigma / return_std)
 - return_std = training_future_ret.std()
 - Confidence is derived from predicted variance, not from the magnitude of μ

 # stocks by sector
    Tech
    AAPL, MSFT, GOOGL, META, NVDA, AMD, INTC, CSCO, ORCL, IBM, ADBE, CRM, INTU, SHOP
    Financials
    JPM, BAC, WFC, GS, MS, V, MA, AXP, PYPL
    Consumer Discretionary
    AMZN, TSLA, MCD, NKE, SBUX, TGT, HD, LOW
    Consumer Staples
    KO, PEP, WMT, MRK (actually pharma but defensive), COST (if you have it)
    Healthcare/Pharma
    JNJ, MRK, ABT, UNH, TMO, PFE
    Energy
    XOM, CVX, COP, SLB
    Industrials / Other
    BRK-B, DIS, NFLX, LYFT, UBER, C (Citigroup)