# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st

# %%
df = pd.read_csv("Fortune_500_Companies.csv")

# %%

st.title("Effect of COVID-19 on Fortune 500 Companies")
st.subheader("Data Exploration")

# %%
covid_years = [2021, 2022]
# Filter the dataframe to include companies that were in the Fortune 500 list in 2020 and 2021
covid_companies = df[df["year"].isin(covid_years)]["name"].unique()

# %%
df = df[df["name"].isin(covid_companies)]
st.write("An overview of the data is given below.")
df

# %%
precovid_years = [2016, 2017, 2018, 2019, 2020]
df_filtered = df[df["year"].isin(precovid_years + covid_years)]

# %%
# df_filtered.columns.to_list()

# %%


# %%
# df_filtered.describe()

# %%
import pandas as pd
import altair as alt

st.write("Let's take a look at the top 10 companies by revenue for each year.")
# Ensure the year column is integer type for proper filtering
df_filtered["year"] = df_filtered["year"].astype(int)

# Create a filtered DataFrame for top 10 companies by revenue for each year
top_10_per_year = (
    df_filtered.sort_values(["year", "revenue_mil"], ascending=[True, False])
    .groupby("year")
    .head(10)
)

# Create a slider selection for the year
slider = alt.binding_range(
    min=top_10_per_year["year"].min(),
    max=top_10_per_year["year"].max(),
    step=1,
    name="Year: ",
)
select_year = alt.selection_single(fields=["year"], bind=slider, value=2016)

# Create the Altair chart
chart = (
    alt.Chart(top_10_per_year)
    .mark_bar()
    .encode(
        x=alt.X(
            "revenue_mil:Q", title="Revenue (in millions)", axis=alt.Axis(grid=False)
        ),
        y=alt.Y("name:N", sort="-x", title=None),
        color="industry:N",
        tooltip=[
            alt.Tooltip("name:N", title="Company Name"),
            alt.Tooltip("revenue_mil:Q", title="Revenue (in millions)"),
            alt.Tooltip("industry:N", title="Industry"),
        ],
    )
    .add_selection(select_year)
    .transform_filter(select_year)
    .properties(
        width=800, height=400, title="Top 10 Companies by Revenue for Each Year"
    )
    .configure_view(
        stroke=None,
    )
)

# Display the chart
chart

st.write(
    "When sliding the slider across, we see that the top 10 companies change dramatically over the years before and after covid. Notice that the Walmart stays at the top, while motor companies like ford and GM move out of the top 10 and healthcare companies move up post covid."
)


# %%
pre_covid = df_filtered[df_filtered["year"].isin(precovid_years)]
post_covid = df_filtered[df_filtered["year"].isin(covid_years)]
pre_covid_grouped = (
    pre_covid.groupby("name")
    .agg({"profit_mil": "mean", "revenue_mil": "mean"})
    .reset_index()
)
post_covid_grouped = (
    post_covid.groupby("name")
    .agg({"profit_mil": "mean", "revenue_mil": "mean"})
    .reset_index()
)


# %%
pre_covid_grouped.rename(
    columns={
        "profit_mil": "mean_profit_precovid",
        "revenue_mil": "mean_revenue_precovid",
    },
    inplace=True,
)
post_covid_grouped.rename(
    columns={"profit_mil": "mean_profit_covid", "revenue_mil": "mean_revenue_covid"},
    inplace=True,
)
merged_df = pre_covid_grouped.merge(post_covid_grouped, on="name", how="inner")
st.write(
    "To gain a sense of the marco trend, let's look at the disribution of revenue before and after covid."
)
# %%
source = merged_df.melt(
    id_vars="name",
    value_vars=["mean_revenue_precovid", "mean_revenue_covid"],
    var_name="type",
    value_name="amount",
)

# Density plot for Pre-COVID
density_pre_covid = (
    alt.Chart(source)
    .transform_density(
        "amount",
        as_=["amount", "density"],
        extent=[0, max(source["amount"]) / 2],
        groupby=["type"],
    )
    .transform_filter(alt.datum.type == "mean_revenue_precovid")
    .mark_area(color="steelblue", opacity=0.5)
    .encode(
        alt.X("amount:Q", title="Revenue (in millions)", axis=alt.Axis(grid=False)),
        alt.Y("density:Q", title="Density", axis=alt.Axis(grid=False)),
    )
    .properties(title="Revenue Distribution Pre-COVID")
)

# Density plot for During COVID
density_covid = (
    alt.Chart(source)
    .transform_density(
        "amount",
        as_=["amount", "density"],
        extent=[0, max(source["amount"]) / 2],
        groupby=["type"],
    )
    .transform_filter(alt.datum.type == "mean_revenue_covid")
    .mark_area(color="orange", opacity=0.5)
    .encode(
        alt.X("amount:Q", title="Revenue (in millions)", axis=alt.Axis(grid=False)),
        alt.Y("density:Q", title="Density", axis=alt.Axis(grid=False)),
    )
    .properties(title="Revenue Distribution During COVID")
)

# Combine the plots
density_plot = alt.hconcat(density_pre_covid, density_covid)

density_plot

st.write(
    "Comparing the revenue distribution before and after covid, we see that the distribution is almost the same, except the maximum density has reduced by a small margin, which tells us there wasn't much change in the overall revenue of all the companies on the whole. "
)

# %%
top_10 = (
    df_filtered.groupby("name")
    .agg({"revenue_mil": "mean"})
    .sort_values("revenue_mil", ascending=False)
    .head(7)
    .reset_index()
)
top_10_companies = top_10["name"].to_list()
top_10_df = df_filtered[df_filtered["name"].isin(top_10_companies)]

line_chart = (
    alt.Chart(top_10_df)
    .mark_line()
    .encode(
        x=alt.X(
            "year:O", title="Year", axis=alt.Axis(labelAngle=0)
        ),  # Adjust the domain to the desired range
        y=alt.Y("revenue_mil:Q", title="Revenue (in millions)"),
        color="name:N",
        tooltip=["name", "year", "revenue_mil"],
    )
    .properties(
        width=500, height=400, title="Revenue Trends of Top 10 Companies Over the Years"
    )
)
line_chart

st.write(
    "Upon examining the revenue trends of the top 10 companies over the years, we see that the revenue of the top companies has been increasing over the years, however there is a cinch in the growth around 2021, which is during the covid period. This could be due to the pandemic and the lockdowns that were imposed, which affected the revenue of the companies."
)


# %%
# Function to calculate Average Annual Growth Rate
def calculate_aagr(df):
    df = df.sort_values(by="year")
    df["profit_growth"] = df["profit_mil"].pct_change()
    aagr = df["profit_growth"].mean(skipna=True)
    return aagr


# Calculate AAGR for each company pre-COVID
pre_covid_aagr = pre_covid.groupby("name").apply(calculate_aagr).reset_index()
pre_covid_aagr.columns = ["name", "pre_covid_aagr"]

# Calculate AAGR for each company post-COVID
post_covid_aagr = post_covid.groupby("name").apply(calculate_aagr).reset_index()
post_covid_aagr.columns = ["name", "post_covid_aagr"]

# Merge the results
aagr_comparison = pd.merge(pre_covid_aagr, post_covid_aagr, on="name", how="inner")
# aagr_comparison

# %%
covid_comparison = merged_df.merge(aagr_comparison, on="name", how="inner")
# covid_comparison

# %%
covid_comparison.dropna(inplace=True)

# %%
# covid_comparison

# %%
covid_comparison["profit_diff"] = (
    covid_comparison["mean_profit_covid"] - covid_comparison["mean_profit_precovid"]
)
covid_comparison["revenue_diff"] = (
    covid_comparison["mean_revenue_covid"] - covid_comparison["mean_revenue_precovid"]
)
covid_comparison["aagr_diff"] = (
    covid_comparison["post_covid_aagr"] - covid_comparison["pre_covid_aagr"]
)

# %%
df_filtered["company_size"] = pd.qcut(
    df_filtered["employees"], q=3, labels=["small", "medium", "large"]
)
# df_filtered

# %%

df_filtered_2020 = df_filtered[df_filtered["year"] == 2020]
# Drop duplicates while keeping the most recent information before covid hit
company_info = df_filtered_2020[
    [
        "name",
        "industry",
        "sector",
        "headquarters_state",
        "headquarters_city",
        "company_size",
        "founder_is_ceo",
        "female_ceo",
        "newcomer_to_fortune_500",
    ]
].drop_duplicates(subset="name")

# Create additional columns
company_info["ceo_female"] = company_info["female_ceo"].apply(
    lambda x: 0 if x == "no" else 1
)
company_info["newcomer"] = company_info["newcomer_to_fortune_500"].apply(
    lambda x: 0 if x == "no" else 1
)
company_info["ceo_founder"] = company_info["founder_is_ceo"].apply(
    lambda x: 0 if x == "no" else 1
)

# Drop the original columns
company_info.drop(
    columns=["founder_is_ceo", "female_ceo", "newcomer_to_fortune_500"], inplace=True
)


# %%
# company_info.info()

# %%
final_df = covid_comparison.merge(company_info, on="name", how="inner")
# final_df.head()

st.write(
    "For the purpose of this analysis, we will focus on three factors: profit difference, revenue difference, and AAGR difference before and after COVID. AAGR refers to the average annual growth rate, which is the average rate at which a company's profit grows over a period of time. We will also consider the company size, sector, and state of headquarters."
)
# %%
profit_diff_plot = sns.histplot(data=final_df, x="profit_diff", kde=True)
plt.title("Distribution of Change in Profit from Pre-COVID to COVID")
plt.xlabel("Change in Profit (in millions)")

st.pyplot(profit_diff_plot.figure)
st.write(
    "The distribution of change in profit from pre-covid to covid is shown above. We see that the distribution is almost normal, with a few outliers on the negative side, which means that most companies have seen a positive change in profit during the covid period."
)
revenue_diff_plot = sns.histplot(data=final_df, x="revenue_diff", kde=True)
plt.title("Distribution of Revenue Difference Between Pre and Post COVID")
plt.xlabel("Revenue Difference (in millions)")
st.pyplot(revenue_diff_plot.figure)

# %%
aagr_plot = sns.histplot(data=final_df, x="aagr_diff", kde=True)
plt.title("Distribution of AAGR Difference Between Pre and Post COVID")
plt.xlabel("AAGR Difference")

st.pyplot(aagr_plot.figure)
st.write(
    "Thus we see that the distribution of all the differences is almost normal, suggesting little to no effect from covid on the companies. However, let's explore further."
)

# %%
nearest = alt.selection_point(
    nearest=True, on="mouseover", fields=["amount"], empty=False
)
scatter = (
    alt.Chart(final_df)
    .mark_circle(opacity=1, size=50)
    .encode(
        x=alt.X("revenue_diff:Q", title="Revenue Difference (in millions)"),
        y=alt.Y("name:N", axis=None),
        color=alt.Color("sector:N", legend=None),
        tooltip=["name", "sector", "revenue_diff", "profit_diff"],
    )
    .properties(
        width=600,
        height=400,
        title="Revenue Difference for Companies before and after COVID",
    )
    .add_params(nearest)
)
scatter

st.write(
    "We see that there are quite a few outliers in the scatterplot of the revenue difference, which means that some companies have seen a significant change in revenue during the covid period. Moving the mouse over these points, we realise that extreme outliers to the right, which saw an increase in revenue, were mostly healthcare and technology companies while the ones who saw a decrease in revenue were motor companies and industrials. This can be explained by the fact that healthcare and technology companies were in demand during the pandemic, while motor companies and industrials were not."
)

# %%
nearest = alt.selection_point(
    nearest=True, on="mouseover", fields=["amount"], empty=False
)
scatter = (
    alt.Chart(final_df)
    .mark_circle(opacity=1, size=50)
    .encode(
        x=alt.X(
            "aagr_diff:Q",
            title="Difference in AAGR (percent difference)",
        ),
        y=alt.Y("name:N", axis=None),
        color=alt.Color("company_size:N"),
        tooltip=["name", "sector", "revenue_diff", "profit_diff"],
    )
    .properties(
        width=600,
        height=400,
        title="Difference in AAGR for Companies before and after COVID",
    )
    .add_params(nearest)
)
scatter

st.write(
    "Coming to the AAGR difference, we see that the companies who had long term growth were small companies while medium companies slowed down. "
)
# %%
nearest = alt.selection_point(
    nearest=True, on="mouseover", fields=["amount"], empty=False
)
scatter = (
    alt.Chart(final_df)
    .mark_circle(opacity=1, size=50)
    .encode(
        x=alt.X("profit_diff:Q", title="Profit Difference (in millions)"),
        y=alt.Y("name:N", axis=None),
        color=alt.Color("headquarters_state:N", legend=None),
        tooltip=["name", "sector", "headquarters_state"],
    )
    .properties(
        width=600,
        height=400,
        title="Profit Difference for Companies before and after COVID",
    )
    .add_params(nearest)
)
scatter

st.write(
    "When we examine the profits, we find that the companies which made most profits are technology, financials, healthcare, and energy, while the ones who made the least profits also belong to the same sectors except for technology. This confounding result can mean that while revenue grew for these sectors, the costs incurred also grew due to great demand."
)

# %%
# profit vs revenue diff
nearest = alt.selection(type="single", on="mouseover", fields=["amount"], nearest=True)
profit_revenue_diff = (
    alt.Chart(final_df)
    .mark_circle()
    .encode(
        x=alt.X("revenue_diff:Q", title="Revenue Difference post Covid (in millions)"),
        y=alt.Y("profit_diff:Q", title="Profit Difference post Covid (in millions)"),
        color=alt.Color("sector:N", legend=None),
        tooltip=["name", "sector", "revenue_diff", "profit_diff"],
    )
    .properties(
        width=600,
        height=400,
        title="Profit vs Revenue Difference for Companies During COVID",
    )
    .add_selection(nearest)
)
profit_revenue_diff

st.write(
    "Looking at this scatterplot, our previous assumptions are confirmed. We can see that technology sector grew in both revenue and profit while healthcare grew in revenue but not profit as they also incurred great costs due to the pandemic."
)

# %%
corr = final_df[
    [
        "revenue_diff",
        "profit_diff",
        "aagr_diff",
        "newcomer",
        "ceo_female",
        "ceo_founder",
    ]
].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

# %%
final_df["profit_diff_norm"] = (
    final_df["profit_diff"] - final_df["profit_diff"].mean()
) / final_df["profit_diff"].std()
final_df["revenue_diff_norm"] = (
    final_df["revenue_diff"] - final_df["revenue_diff"].mean()
) / final_df["revenue_diff"].std()
final_df["aagr_diff_norm"] = (
    final_df["aagr_diff"] - final_df["aagr_diff"].mean()
) / final_df["aagr_diff"].std()

# %%
final_df["composite_score"] = (
    final_df["profit_diff_norm"] + final_df["revenue_diff_norm"]
)

# %%
min_score = final_df["composite_score"].min()
max_score = final_df["composite_score"].max()
final_df["impact_score"] = (
    10 * (final_df["composite_score"] - min_score) / (max_score - min_score)
)

"Now using a composite score calculated from the normalized values of the profit and revenue differences, we can rank the companies based on their impact score. The impact score is a measure of how much a company was affected by the pandemic. The higher the score, the more the company benefitted, and the lower the score, the more the company was affected negatively."
# %%
top_10_gains = final_df.nlargest(10, "composite_score")
bar = (
    alt.Chart(top_10_gains)
    .mark_bar()
    .encode(
        x=alt.X("composite_score:Q", title="Impact Score", axis=alt.Axis(grid=False)),
        y=alt.Y("name:N", title=None, sort="-x", axis=alt.Axis(titlePadding=20)),
        color=alt.Color("sector:N", legend=None),
        tooltip=["name", "sector", "composite_score"],
    )
    .properties(
        width=300, height=250, title="Top 10 Companies positively impacted by COVID"
    )
)
top_10_losses = final_df.nsmallest(10, "composite_score")
bar2 = (
    alt.Chart(top_10_losses)
    .mark_bar()
    .encode(
        x=alt.X("composite_score:Q", title="Impact Score", axis=alt.Axis(grid=False)),
        y=alt.Y("name:N", title=None, sort="-x", axis=alt.Axis(titlePadding=20)),
        color=alt.Color("sector:N", legend=None),
        tooltip=["name", "sector", "composite_score"],
    )
    .properties(
        width=300, height=250, title="Top 10 Companies negatively impacted by COVID"
    )
)
bar | bar2

# %%
st.write(
    "The bar chart below shows the comparison of the top 5 sectors least impacted by COVID and the top 5 sectors most impacted by COVID. Looking at this confirms our assumption of healthcare, technology and retail companies benefitting most, while industrial and motor companies went through a rough patch."
)
sector_grouped = (
    final_df.groupby("sector")
    .agg({"composite_score": "mean"})
    .sort_values("composite_score", ascending=False)
)
sector_grouped.reset_index(inplace=True)
sector_gains = sector_grouped.nlargest(4, "composite_score")
sector_losses = sector_grouped.nsmallest(4, "composite_score")
bar = (
    alt.Chart(sector_gains)
    .mark_bar(color="darkblue")
    .encode(
        x=alt.X("composite_score:Q", title="Impact Score"),
        y=alt.Y("sector:N", title=None, sort="-x", axis=alt.Axis(titlePadding=20)),
        tooltip=["sector", "composite_score"],
    )
    .properties(width=250, height=200, title="Top 5 sectors least impacted by COVID")
)
bar2 = (
    alt.Chart(sector_losses)
    .mark_bar(color="orange")
    .encode(
        x=alt.X("composite_score:Q", title="Impact Score", axis=alt.Axis(grid=False)),
        y=alt.Y("sector:N", title=None, sort="-x", axis=alt.Axis(titlePadding=20)),
        tooltip=["sector", "composite_score"],
    )
    .properties(width=250, height=200, title="Top 5 sectors most impacted by COVID")
)
bar | bar2
