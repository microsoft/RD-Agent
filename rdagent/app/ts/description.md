
# Introduction

This task involves predicting cryptocurrency prices. The data you need is in the `data` folder. Your goal is to create a trading algorithm that uses historical data to forecast future prices. At this stage, it is unclear which targets can be predicted.

You will examine the data to identify predictable targets and build a suitable dataset. Then, you will develop a data-driven solution based on your findings.

The prediction target is flexible and can be based on any metric. You may use any number of instruments, and you can exclude those that are difficult to predict. For example, you could focus on predicting the excess return of certain instruments compared to BTC. The prediction horizon is also open and can be set to any time frame. Your main goal is to find patterns in the data that are easy to model and from which we can make a profit with the signal (e.g. the return in the coming 5 minutes).

You can experiment with different targets and assess how predictable they are. To ensure that your results are easy to compare, use consistent metrics such as accuracy or the correlation between your predicted values and the actual targets.

Here is the list of available instruments:
1000000BOBUSDT ARCUSDT BROCCOLI714USDT DOGSUSDT GRASSUSDT KNCUSDT NEARUSDT PUFFERUSDT SONICUSDT UNIUSDT 1000000MOGUSDT ARKMUSDT BROCCOLIF3BUSDT DOLOUSDT GRIFFAINUSDT KOMAUSDT NEIROETHUSDT PUMPBTCUSDT SOONUSDT USDCUSDT 1000BONKUSDT ARKUSDT BRUSDT DOODUSDT GRTUSDT KSMUSDT NEIROUSDT PUMPUSDT SOPHUSDT USTCUSDT 1000BTTCUSDT ARPAUSDT BSVUSDT DOTECOUSDT GTCUSDT LAUSDT NEOUSDT PUNDIXUSDT SPELLUSDT USUALUSDT 1000CATUSDT ARUSDT BSWUSDT DOTUSDT GUNUSDT LAYERUSDT NFPUSDT PYTHUSDT SPXUSDT UXLINKUSDT 1000CHEEMSUSDT ASRUSDT BTCDOMUSDT DRIFTUSDT GUSDT LDOUSDT NILUSDT QNTUSDT SQDUSDT VANAUSDT 1000FLOKIUSDT ASTRUSDT BTCSTUSDT DUSDT HAEDALUSDT LENDUSDT NKNUSDT QTUMUSDT SRMUSDT VANRYUSDT 1000LUNCUSDT ATAUSDT BTCUSDT DUSKUSDT HBARUSDT LEVERUSDT NMRUSDT QUICKUSDT SSVUSDT VELODROMEUSDT 1000PEPEUSDT ATHUSDT BTSUSDT DYDXUSDT HEIUSDT LINAUSDT NOTUSDT RADUSDT STEEMUSDT VETUSDT 1000RATSUSDT ATOMUSDT BTTUSDT DYMUSDT HFTUSDT LINKUSDT NTRNUSDT RAREUSDT STGUSDT VICUSDT 1000SATSUSDT AUCTIONUSDT BUSDT EDUUSDT HIFIUSDT LISTAUSDT NULSUSDT RAYSOLUSDT STMXUSDT VIDTUSDT 1000SHIBUSDT AUDIOUSDT BZRXUSDT EGLDUSDT HIGHUSDT LITUSDT NUUSDT RAYUSDT STORJUSDT VINEUSDT 1000WHYUSDT AUSDT C98USDT EIGENUSDT HIPPOUSDT LOKAUSDT NXPCUSDT RDNTUSDT STOUSDT VIRTUALUSDT 1000XECUSDT AVAAIUSDT CAKEUSDT ENAUSDT HIVEUSDT LOOMUSDT OBOLUSDT REDUSDT STPTUSDT VOXELUSDT 1000XUSDT AVAUSDT CATIUSDT ENJUSDT HMSTRUSDT LPTUSDT OCEANUSDT REEFUSDT STRAXUSDT VTHOUSDT 1INCHUSDT AVAXUSDT CELOUSDT ENSUSDT HNTUSDT LQTYUSDT OGNUSDT REIUSDT STRKUSDT VVVUSDT 1MBABYDOGEUSDT AWEUSDT CELRUSDT EOSUSDT HOMEUSDT LRCUSDT OGUSDT RENDERUSDT STXUSDT WALUSDT AAVEUSDT AXLUSDT CETUSUSDT EPICUSDT HOOKUSDT LSKUSDT OMGUSDT RENUSDT SUIUSDT WAVESUSDT ACEUSDT AXSUSDT CFXUSDT EPTUSDT HOTUSDT LTCUSDT OMNIUSDT RESOLVUSDT SUNUSDT WAXPUSDT ACHUSDT B2USDT CGPTUSDT ETCUSDT HUMAUSDT LUMIAUSDT OMUSDT REZUSDT SUPERUSDT WCTUSDT ACTUSDT B3USDT CHESSUSDT ETHFIUSDT HYPERUSDT LUNA2USDT ONDOUSDT RIFUSDT SUSDT WIFUSDT ACXUSDT BABYUSDT CHILLGUYUSDT ETHUSDT HYPEUSDT LUNAUSDT ONEUSDT RLCUSDT SUSHIUSDT WLDUSDT ADAUSDT BADGERUSDT CHRUSDT ETHWUSDT ICXUSDT MAGICUSDT ONGUSDT RNDRUSDT SWARMSUSDT WOOUSDT AEROUSDT BAKEUSDT CHZUSDT FARTCOINUSDT IDEXUSDT MANAUSDT ONTUSDT RONINUSDT SWELLUSDT WUSDT AEVOUSDT BALUSDT CKBUSDT FETUSDT IDUSDT MANTAUSDT OPUSDT ROSEUSDT SXPUSDT XAIUSDT AGIXUSDT BANANAS31USDT COCOSUSDT FHEUSDT ILVUSDT MASKUSDT ORBSUSDT RPLUSDT SXTUSDT XCNUSDT AGLDUSDT BANANAUSDT COMBOUSDT FILUSDT IMXUSDT MATICUSDT ORCAUSDT RSRUSDT SYNUSDT XEMUSDT AGTUSDT BANDUSDT COMPUSDT FIOUSDT INITUSDT MAVUSDT ORDIUSDT RUNEUSDT SYRUPUSDT XLMUSDT AI16ZUSDT BANKUSDT COOKIEUSDT FISUSDT INJUSDT MBLUSDT OXTUSDT RVNUSDT SYSUSDT XMRUSDT AIOTUSDT BANUSDT COSUSDT FLMUSDT IOSTUSDT MBOXUSDT PARTIUSDT SAFEUSDT TAIKOUSDT XRPUSDT AIUSDT BATUSDT COTIUSDT FLOWUSDT IOTAUSDT MDTUSDT PAXGUSDT SAGAUSDT TAOUSDT XTZUSDT AIXBTUSDT BBUSDT COWUSDT FLUXUSDT IOTXUSDT MELANIAUSDT PENDLEUSDT SANDUSDT THETAUSDT XVGUSDT AKROUSDT BCHUSDT CRVUSDT FOOTBALLUSDT IOUSDT MEMEFIUSDT PENGUUSDT SANTOSUSDT THEUSDT XVSUSDT AKTUSDT BDXNUSDT CTSIUSDT FORMUSDT IPUSDT MEMEUSDT PEOPLEUSDT SCRTUSDT TIAUSDT YFIIUSDT ALCHUSDT BEAMXUSDT CVXUSDT FORTHUSDT JASMYUSDT MERLUSDT PERPUSDT SCRUSDT TNSRUSDT YFIUSDT ALGOUSDT BELUSDT CYBERUSDT FRONTUSDT JELLYJELLYUSDT METISUSDT PHAUSDT SCUSDT TOKENUSDT YGGUSDT ALICEUSDT BERAUSDT DARUSDT FTMUSDT JOEUSDT MEUSDT PHBUSDT SEIUSDT TOMOUSDT ZECUSDT ALPACAUSDT BICOUSDT DASHUSDT FTTUSDT JSTUSDT MEWUSDT PIPPINUSDT SFPUSDT TONUSDT ZENUSDT ALPHAUSDT BIDUSDT DEEPUSDT FUNUSDT JTOUSDT MILKUSDT PIXELUSDT SHELLUSDT TRBUSDT ZEREBROUSDT ALPINEUSDT BIGTIMEUSDT DEFIUSDT FXSUSDT JUPUSDT MINAUSDT PLUMEUSDT SIGNUSDT TROYUSDT ZETAUSDT ALTUSDT BIOUSDT DEGENUSDT GALAUSDT KAIAUSDT MKRUSDT PNUTUSDT SIRENUSDT TRUMPUSDT ZILUSDT AMBUSDT BLUEBIRDUSDT DEGOUSDT GALUSDT KAITOUSDT MLNUSDT POLUSDT SKATEUSDT TRUUSDT ZKJUSDT ANCUSDT BLURUSDT DENTUSDT GASUSDT KASUSDT MOCAUSDT POLYXUSDT SKLUSDT TRXUSDT ZKUSDT ANIMEUSDT BLZUSDT DEXEUSDT GHSTUSDT KAVAUSDT MOODENGUSDT PONKEUSDT SKYAIUSDT TSTUSDT ZROUSDT ANKRUSDT BMTUSDT DFUSDT GLMRUSDT KDAUSDT MORPHOUSDT POPCATUSDT SLERFUSDT TURBOUSDT ZRXUSDT ANTUSDT BNBUSDT DGBUSDT GLMUSDT KEEPUSDT MOVEUSDT PORT3USDT SLPUSDT TUSDT APEUSDT BNTUSDT DIAUSDT GMTUSDT KERNELUSDT MOVRUSDT PORTALUSDT SNTUSDT TUTUSDT API3USDT BOMEUSDT DODOUSDT GMXUSDT KEYUSDT MTLUSDT POWRUSDT SNXUSDT TWTUSDT APTUSDT BONDUSDT DODOXUSDT GOATUSDT KLAYUSDT MUBARAKUSDT PROMPTUSDT SOLUSDT UMAUSDT ARBUSDT BRETTUSDT DOGEUSDT GPSUSDT KMNOUSDT MYROUSDT PROMUSDT SOLVUSDT UNFIUSDT


## Requirements
- When working with time series data, always split your data based on time. The validation set must only contain data from later dates than the training set, and the test set must come after the validation set. This prevents data leakage and ensures fair model evaluation.
- Make sure your validation data covers a period of more than **12 months** (the validation evaluation in scores.csv should also cover the same length!). This helps ensure that your evaluation of the prediction is reliable and robust. We only need a very short period of test time, e.g., 3 months. Please leave most of the data for training and validation.
- Always print out key details about your dataset, including:
  - The instruments you used
  - The points in time used to split the data
  - The final feature list use for training model.
- When creating the submission files, make sure they have exactly three columns: date, symbol, and pred  
- If you face resource limits, it is better to use fewer instruments but keep a longer time period, rather than just reducing the number of samples. A bad example is limiting the data to only 1 million samples, as this will usually keep only a short time range since the data is typically sorted by time.
- DO NOT CACHE THE DATASET OR THE MODEL! Always create new features and train models from the raw data each time you run the code.

### Tips for feature engineering
- When building features, DO NOT introduce any future information. Data leakage will make the solution invalid and is not allowed. Following are some typical errors that leakeage future information:
- `data[f'{col_mean}_norm'] = data[col_mean] / data.groupby(['symbol', 'date'])[col_mean].transform('mean')` uses information from the end of the day to normalize data, but that information is not available at that time point. Instead, you can use rolling statistics for normalization.
```Python
rolling_mean_1d = (
    data.groupby('symbol', group_keys=False)
        .apply(lambda grp: grp.set_index('datetime')[col_mean].rolling('1D', closed='both').mean())
        .reset_index(level=0, drop=True)
)
data[f'{col_mean}_norm'] = data[col_mean] / rolling_mean_1d
```


- Make sure your features remain stationary so their distributions stay consistent over time; otherwise, prediction results can be biased.
  - For example:
    - `np.log(df['close']).shift(1).rolling(window=5, min_periods=5).std()` is preferred over `df['close'].shift(1).rolling(window=5, min_periods=5).std()`, as using log prices makes the standard deviation independent of the price level.
    - `df['close'].shift(1).rolling(window=10, min_periods=10).mean() / df['close_day_mean']` is better than just using the mean, since normalizing by the daily mean removes the effect of absolute price units.

### Tips for Creating the Prediction Target
- DO NOT use the absolute price as the target. Instead, choose a target that captures future price changes, such as the price movement over a short period.
- The predictions are intended for trading purposes, so your prediction target should match a tradable action. For example, if you use features at time `T`, and assume it takes `1` minute to enter the position and `1` more minute to close the position after holding it for `h` minutes, the prediction target should reflect the price change from `T + 1` to `T + h + 1`. A typical error is predicting the price change from  `T` to `T + h`,
- DON'T predict volatility!!!!!!!
- When working with multiple symbols, make sure you calculate the target separately for each symbol. If your data includes several symbols combined in one table, use  
  `data['target'] = data.groupby("symbol")["log_price"].transform(lambda s: s.shift(-(h + 1)) - s.shift(-1))`
  rather than
  `data['target'] = data['log_price'].shift(-(h + 1)) - data['log_price'].shift(-1)`,
  so you don't mix prices from different symbols.
