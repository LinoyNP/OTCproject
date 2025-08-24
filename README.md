
# 🚀 OTCproject

A graph-based analysis of user rating behavior in Bitcoin over-the-counter (OTC) trading.
---
## 🧠 What’s this about?
This project was developed as part of a Social Network Analysis course.

It explores whether voting bias exists in peer ratings — specifically, whether users who receive positive feedback are more likely to give positive ratings themselves (and vice versa for negative).
Using a real-world dataset of Bitcoin OTC trades, we constructed a user-to-user rating graph and analyzed patterns of reciprocity, clustering, and sentiment bias.

---

## 📊 Key Questions

- Do people tend to "return the favor" in ratings?
- Is there evidence of echo chambers?
---

## 🔍 Technologies Used

- Python 🐍
- NetworkX for graph analysis
- Matplotlib / Seaborn for visualization
- Pandas for data handling

## 📁 Dataset

- The data is taken from the **Bitcoin OTC trust weighted network**, available from [SNAP (Stanford)](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html).
---

![User Rating Graph](https://github.com/LinoyNP/OTCproject/blob/main/p.png?raw=true)

- Example of user rating network.
- blue = Old ratings (up to 2014) red = Relevant ratings (2014–2016),
- ◯ Positive Ratings  ▲ Negative Ratings
