SECTORS_LIST = '''
Basic Materials
Chemicals
Metals & Mining
Construction & Building Materials
Communication Services
Consumer Cyclical
Consumer Defensive
Energy
Financial Services
Banks
Insurance
Healthcare
Pharmaceuticals & Biotechnology
Industrials
Aerospace & Defense
Real Estate
Technology
Semiconductors & Electronics
Utilities
Transportation
'''

SECTOR_PROMPT = """
I will give you a ticker, which may be delisted OR active, however you must classify it in one of the following sectors:

Sectors:
{sectors_list}
ETFs
Bonds

Only return the sector name, NOTHING ELSE

Ticker: {ticker}
"""

FNG_PROMPT = """
You are an expert financial and economic analyst. You are known to make very specific and very accurate predictions using very little data. 

You will be provided the CNN Fear and Greed Index over the last 1 to 12 months. Using the trends, stage in the cycle, and your prediction, especially about what will happen in the short to medium term (3 to 6 months), you need to select which of the following sectors to invest in. Ensure that you only select the most well to perform sector(s), selecting at least one and at most three.

Think through your response so thoroughly such that if I were to ask you this question again with the same data, you'd return the same rationale and therefore the same response.

Return your response as a concatenation of the sectors delimited by comma. Do not return ANY other information. Return PLAIN text. NO encoding. ONLY from the choices.

```
Sectors:
{sectors_list}
```
```
Monthly F&G Data (Oldest to newest): {fng_vals}
```
"""

def get_fng_prompt(fng_vals, sectors_list=SECTORS_LIST):
    return FNG_PROMPT.format(fng_vals=fng_vals, sectors_list=sectors_list)