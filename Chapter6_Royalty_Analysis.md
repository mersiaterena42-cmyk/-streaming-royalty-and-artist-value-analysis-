# Chapter 6: Music Streaming Royalty & Artist Value Analysis

**New concepts introduced:** business-logic calculations (royalty math), per-stream rate modelling, weighted metrics, correlation analysis, percentile ranking, building a single "value score" from multiple signals.

**Prerequisite:** Chapters 1–5 (Pandas basics, filtering/grouping, SQL, dashboards, time series).

**The role this mirrors:** royalty/financial analysts at labels, distributors (DistroKid, Believe, TuneCore), and rights organizations (Kobalt, ASCAP, PRS) spend their time on two core questions: *"how much do we owe this artist?"* and *"how valuable is this artist to us?"* — those are two different calculations, and this chapter builds both.

---

## 6.0 The dataset

We'll work with simulated streaming data, because real per-stream royalty data is commercially confidential — but the structure mirrors what you'd get from a DSP (Digital Service Provider, i.e. Spotify/Apple Music/etc.) reporting feed.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

artists = ["Artist A", "Artist B", "Artist C", "Artist D", "Artist E"]
platforms = ["Spotify", "Apple Music", "YouTube Music", "Amazon Music"]
n = 5000

# One row = one day's aggregated streaming activity for one artist on one platform
df = pd.DataFrame({
    "date": np.random.choice(pd.date_range("2024-01-01", "2024-12-31"), n),
    "artist": np.random.choice(artists, n, p=[0.35, 0.25, 0.2, 0.12, 0.08]),
    "platform": np.random.choice(platforms, n, p=[0.45, 0.25, 0.2, 0.1]),
    "country": np.random.choice(["US", "UK", "DE", "BR", "JP"], n),
    "streams": np.random.poisson(lam=2000, size=n),
    "saves": np.random.poisson(lam=40, size=n),
    "skips": np.random.poisson(lam=300, size=n),
})

df = df.sort_values("date").reset_index(drop=True)
```

**Why simulate instead of just describing the math:** royalty modelling only makes sense when you can see it applied to messy, repeated data across platforms and countries — that's exactly the situation a real analyst works in.

---

## 6.1 Royalty rates differ by platform — model that explicitly

Per-stream payout rates vary by platform, country, and subscription tier in reality. We'll simplify to a per-platform average rate (a reasonable first model — and a defensible simplification to explain in an interview).

```python
# Approximate blended per-stream rates (illustrative, not real published figures)
rate_per_stream = {
    "Spotify": 0.0033,
    "Apple Music": 0.0078,
    "YouTube Music": 0.0020,
    "Amazon Music": 0.0040,
}

# .map() looks up each row's platform in the dictionary and returns the matching rate —
# this is the cleanest way to apply a lookup table across a column
df["rate"] = df["platform"].map(rate_per_stream)

# Royalty owed for that row = streams * rate for that platform
df["royalty_usd"] = df["streams"] * df["rate"]
```

**Why not just use one flat rate for everything?** Platforms negotiate different rates and pass through different shares of subscription/ad revenue. Using a blended single rate would understate Apple Music revenue and overstate YouTube Music revenue — exactly the kind of error a royalty analyst is hired to avoid.

---

## 6.2 Total royalties owed per artist

```python
# Group by artist, sum royalties — this answers "how much do we owe each artist?"
royalties_by_artist = df.groupby("artist")["royalty_usd"].sum().sort_values(ascending=False)
print(royalties_by_artist.round(2))

plt.figure(figsize=(7, 4))
royalties_by_artist.plot(kind="bar", color="#1DB954")  # Spotify green, thematically fitting
plt.title("Total Royalties Owed by Artist (2024)")
plt.ylabel("Royalty (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

This is a **ranking** question with a handful of categories — a sorted bar chart is the right call, same reasoning as Chapter 2.

---

## 6.3 Royalty by platform AND artist — where is the money actually coming from?

```python
# pivot_table is a generalisation of groupby: rows=artist, columns=platform,
# values=summed royalty. aggfunc="sum" is explicit (default is "mean", which
# would silently give you the wrong number here)
royalty_pivot = df.pivot_table(
    index="artist", columns="platform", values="royalty_usd", aggfunc="sum"
)
print(royalty_pivot.round(2))

royalty_pivot.plot(kind="bar", stacked=True, figsize=(8, 5))
plt.title("Royalty Revenue by Artist and Platform")
plt.ylabel("Royalty (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Why this matters for an analyst, not just a chart exercise:** an artist earning $50k entirely from Spotify is in a very different (riskier) position than one earning $50k spread across four platforms — platform concentration is a real risk metric labels track.

---

## 6.4 Quantifying "artist value" — beyond raw revenue

Raw royalty totals tell you *what's been paid*, not *how valuable the artist is going forward*. A real value assessment combines multiple signals:

- **Revenue** — money already earned
- **Engagement quality** — saves per stream (people choosing to keep the song, not just having it autoplay)
- **Retention** — skip rate (low skip rate = people are actually listening)
- **Growth trend** — is monthly streaming volume increasing or declining?

```python
# Build one row per artist with all four signals
artist_metrics = df.groupby("artist").agg(
    total_streams=("streams", "sum"),
    total_royalty=("royalty_usd", "sum"),
    total_saves=("saves", "sum"),
    total_skips=("skips", "sum"),
).reset_index()

# Engagement rate: saves as a proportion of streams — higher is better
artist_metrics["save_rate"] = artist_metrics["total_saves"] / artist_metrics["total_streams"]

# Skip rate: skips as a proportion of streams — lower is better
artist_metrics["skip_rate"] = artist_metrics["total_skips"] / artist_metrics["total_streams"]

print(artist_metrics.round(4))
```

### Growth trend — month-over-month streaming change

```python
df["month"] = df["date"].dt.to_period("M")
monthly = df.groupby(["artist", "month"])["streams"].sum().reset_index()

# pct_change() compares each month to the previous one, per artist —
# groupby().transform() applies the calculation within each artist's group
# separately, so Artist B's growth isn't compared against Artist A's numbers
monthly["growth_pct"] = monthly.groupby("artist")["streams"].pct_change()

# Average month-over-month growth rate per artist, ignoring the first month
# of each group (which has no prior month to compare to, so it's NaN)
avg_growth = monthly.groupby("artist")["growth_pct"].mean()
artist_metrics = artist_metrics.merge(
    avg_growth.rename("avg_monthly_growth"), on="artist"
)
print(artist_metrics[["artist", "avg_monthly_growth"]].round(4))
```

### Combining signals into a single value score

```python
# Min-max normalise each metric to a 0-1 scale so they're comparable —
# without this, total_royalty (in the thousands) would completely dominate
# save_rate (a small decimal), regardless of actual importance
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

artist_metrics["royalty_score"] = normalize(artist_metrics["total_royalty"])
artist_metrics["engagement_score"] = normalize(artist_metrics["save_rate"])
artist_metrics["retention_score"] = normalize(1 - artist_metrics["skip_rate"])  # invert: lower skips = higher score
artist_metrics["growth_score"] = normalize(artist_metrics["avg_monthly_growth"])

# Weighted composite — weights reflect business priorities, not a fixed rule.
# This is a judgment call you should be ready to defend in an interview.
artist_metrics["value_score"] = (
    0.40 * artist_metrics["royalty_score"] +
    0.20 * artist_metrics["engagement_score"] +
    0.15 * artist_metrics["retention_score"] +
    0.25 * artist_metrics["growth_score"]
)

ranked = artist_metrics.sort_values("value_score", ascending=False)
print(ranked[["artist", "value_score"]].round(3))

plt.figure(figsize=(7, 4))
plt.barh(ranked["artist"], ranked["value_score"], color="#1DB954")
plt.xlabel("Composite Value Score (0-1)")
plt.title("Artist Value Ranking")
plt.gca().invert_yaxis()  # highest-ranked artist at the top
plt.tight_layout()
plt.show()
```

**Why weight growth at 0.25, almost as much as current royalty (0.40)?** Because a label's real question is forward-looking: "who should we re-sign or invest in?" — not just "who made the most money last year." An artist with strong growth but lower current revenue (e.g. someone breaking out) can be more valuable going forward than someone large but flat or declining. This is the kind of reasoning — not the exact numbers — an interviewer wants to hear you articulate.

---

## 6.5 Does engagement actually predict revenue?

```python
# Scatter plot: is there a real relationship between save_rate and royalty,
# or are they unrelated signals? This justifies (or challenges) including
# both in the value score above.
plt.figure(figsize=(6, 5))
plt.scatter(artist_metrics["save_rate"], artist_metrics["total_royalty"], s=80, color="#1DB954")
for _, row in artist_metrics.iterrows():
    plt.annotate(row["artist"], (row["save_rate"], row["total_royalty"]))
plt.xlabel("Save Rate")
plt.ylabel("Total Royalty (USD)")
plt.title("Engagement vs Revenue")
plt.tight_layout()
plt.show()

# Quantify the relationship numerically, not just visually
correlation = artist_metrics["save_rate"].corr(artist_metrics["total_royalty"])
print(f"Correlation between save rate and royalty: {correlation:.2f}")
```

This is a **relationship between two continuous variables** — a scatter plot is the right chart, straight from Chapter 1's framework, and `.corr()` gives you the number to back up what the chart shows.

---

## 6.6 Exercises

1. Recalculate `royalty_pivot` using `aggfunc="mean"` instead of `"sum"` and explain in a sentence why the result would be misleading if used to answer "how much do we owe this artist."
2. Add a `country` breakdown: which country generates the most royalty revenue per platform?
3. Recompute the `value_score` with growth weighted at 0.40 and royalty at 0.25. Does the artist ranking change? Why might a label choose these weights instead?
4. Build a "platform concentration" metric — what percentage of each artist's total royalty comes from their single biggest platform? Flag any artist above 70%.
5. **Challenge:** identify which artist had the single best month (highest `growth_pct`) across the whole year, and which platform drove it.

---

## 6.7 Interview-style questions

**Q1: Why did you normalize the metrics before combining them into a value score?**
> Discussion point: without normalization, a metric measured in thousands of dollars would dominate a metric measured as a small decimal (like save rate), regardless of how important each one actually is. Normalizing puts them on the same scale so the *weights* — not the units — control the outcome.

**Q2: Your value score weights are 40/20/15/25. How would you defend those numbers if a stakeholder pushed back?**
> Discussion point: there's no single correct answer — show that you understand they're a business judgment call, ideally validated against a known outcome (e.g. "do artists we previously identified as high-value score highly here?") rather than asserted from intuition alone.

**Q3: How would your royalty calculation change if a platform paid based on listener subscription tier (free/ad-supported vs premium) instead of a flat per-stream rate?**
> Discussion point: you'd need tier as an additional column and a rate table keyed on platform *and* tier, then a more granular `.map()`/merge rather than a single dictionary lookup — show you understand the model scales by adding dimensions, not by rewriting the logic.

**Q4: If you found that save rate and royalty were strongly correlated, would you drop one of them from the value score?**
> Discussion point: not necessarily — correlation tells you they move together, not that one is redundant or causal. Worth distinguishing correlation from causation here, and noting you'd want more data (or a longer time window) before simplifying the model.

**Q5: What's a risk in using only 12 months of data to calculate "average monthly growth"?**
> Discussion point: seasonality (e.g. a holiday hit single skewing one month), one outlier month dominating the average, and survivorship bias if struggling artists are dropped from the catalogue before the analysis window starts.

---

**Where this fits in your portfolio:** this project is the strongest one to lead with if you're applying to labels, distributors, or music-tech analyst roles — it's the only chapter that combines royalty/financial logic with a genuine "build a metric" judgment call, which is exactly what gets discussed in case-study interviews.
