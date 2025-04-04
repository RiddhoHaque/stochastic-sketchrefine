SELECT PACKAGE(*) AS P
FROM Stock_Investments_Half
SUCH THAT
SUM(Price) <= 500 AND
SUM(Gain) >= 350 WITH PROBABILITY >= 0.95
MAXIMIZE EXPECTED SUM(Gain)