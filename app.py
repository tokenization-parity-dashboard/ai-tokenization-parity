
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Tokenization Parity Dashboard",
    layout="wide"
)

st.title("AI Tokenization Parity Dashboard")

st.write(
    "FLORES 병렬 문장을 이용하여 Baseline BPE tokenizer와 "
    "Temperature Sampling(alpha=0.3)을 적용해 학습한 tokenizer의 "
    "언어별 Tokenization Parity를 비교합니다."
)

# 실험 결과
data = {
    "Language": [
        "English", "Korean", "Japanese", "Chinese",
        "Spanish", "French", "Arabic", "Hindi"
    ],
    "Baseline Parity": [
        1.00, 1.95, 1.50, 1.26,
        1.45, 1.50, 1.81, 3.24
    ],
    "Temperature Parity": [
        1.00, 1.22, 1.17, 1.05,
        1.29, 1.34, 1.32, 2.58
    ]
}

df = pd.DataFrame(data)

# 영어 기준과의 격차
df["Baseline Gap"] = abs(df["Baseline Parity"] - 1.0)
df["Temperature Gap"] = abs(df["Temperature Parity"] - 1.0)

# 격차 감소율
df["Gap Reduction (%)"] = 0.0

for i in range(len(df)):
    if df.loc[i, "Baseline Gap"] != 0:
        df.loc[i, "Gap Reduction (%)"] = (
            (df.loc[i, "Baseline Gap"] -
             df.loc[i, "Temperature Gap"])
            / df.loc[i, "Baseline Gap"]
            * 100
        )

# -------------------------
# 전체 결과
# -------------------------

st.header("1. Overall Results")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Baseline Mean Gap",
    "0.816"
)

col2.metric(
    "After Temperature Sampling",
    "0.424"
)

col3.metric(
    "Mean Gap Reduction",
    "48.0%"
)

# -------------------------
# 언어 선택
# -------------------------

st.header("2. Language Comparison")

selected_language = st.selectbox(
    "Select a language",
    df["Language"]
)

selected = df[
    df["Language"] == selected_language
].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Baseline Parity",
    f"{selected['Baseline Parity']:.2f}"
)

col2.metric(
    "Temperature Parity",
    f"{selected['Temperature Parity']:.2f}"
)

if selected_language == "English":
    col3.metric(
        "Gap Reduction",
        "Reference"
    )
else:
    col3.metric(
        "Gap Reduction",
        f"{selected['Gap Reduction (%)']:.1f}%"
    )

# -------------------------
# 전체 언어 그래프
# -------------------------

st.header("3. Tokenization Parity Across Languages")

fig, ax = plt.subplots(figsize=(10, 5))

x = range(len(df))

ax.plot(
    x,
    df["Baseline Parity"],
    marker="o",
    label="Baseline"
)

ax.plot(
    x,
    df["Temperature Parity"],
    marker="o",
    label="Temperature Sampling"
)

ax.axhline(
    1.0,
    linestyle="--",
    label="English baseline (1.0)"
)

ax.set_xticks(list(x))
ax.set_xticklabels(df["Language"])

ax.set_xlabel("Language")
ax.set_ylabel("Tokenization Parity")

ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# -------------------------
# 결과표
# -------------------------

st.header("4. Detailed Results")

display_df = df.copy()

display_df["Gap Reduction (%)"] = (
    display_df["Gap Reduction (%)"].round(1)
)

st.dataframe(
    display_df,
    use_container_width=True
)

# -------------------------
# 해석
# -------------------------

st.header("5. Interpretation")

st.write(
    "In this experiment, temperature-based sampling reduced "
    "the average tokenization parity gap across the seven "
    "non-English languages from 0.816 to 0.424, corresponding "
    "to a 48.0% reduction."
)

st.write(
    "The result suggests that increasing the relative representation "
    "of lower-resource languages during tokenizer training can reduce "
    "tokenization disparity, although the degree of improvement differs "
    "across languages."
)
