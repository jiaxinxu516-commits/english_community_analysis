import streamlit as st
import pandas as pd
import plotly.express as px
import json
import collections
import re
import os

# 1. 核心环境修复：确保国内下载大模型不卡顿
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import pipeline

# 2. 页面全局配置
st.set_page_config(layout="wide", page_title="Rokid Reddit AI Dashboard")

# 英文停用词表
STOPWORDS = set(["the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "for", "in", "on", "at", "by", "with", "of", "it", "its", "this", "that", "i", "you", "he", "she", "they", "we", "my", "your", "have", "has", "can", "will", "do", "not", "just", "so", "if", "about", "like", "any", "get", "me", "from", "up", "out", "how", "what", "when", "some", "more", "there", "would", "am", "or", "out", "want", "one", "know", "has", "been", "me", "dont", "think", "anyone", "even", "still"] + [str(i) for i in range(10)])

# 3. 初始化大模型
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

# 4. 读取数据并进行多维度时间特征提取
@st.cache_data
def load_and_analyze_data():
    file_path = "rokid_official_full_data2.json"
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    sentiment_task = load_sentiment_pipeline()
    flattened_list = []
    
    # 这里默认跑前 600 条。如果想跑全量，可以把 [:600] 删掉
    for item in raw_data:
        meta = item.get("meta", {})
        content = item.get("content", {})
        stats = item.get("stats", {})
        author = item.get("author", {})
        
        title = content.get("title", "")
        selftext = content.get("selftext", "")
        full_text = f"{title}. {selftext}"[:512]
        
        sentiment_label = "neutral"
        if full_text.strip():
            try:
                res = sentiment_task(full_text)[0]
                sentiment_label = res['label']
            except:
                pass

        row = {
            "id": meta.get("id"),
            "created_str": meta.get("created_str"),
            "title": title,
            "selftext": selftext,
            "score": stats.get("score", 0),
            "upvote_ratio": stats.get("upvote_ratio", 1.0),
            "num_comments": stats.get("num_comments", 0),
            "author_name": author.get("name", "Unknown"),
            "sentiment": sentiment_label
        }
        flattened_list.append(row)
        
    df = pd.DataFrame(flattened_list)
    
    # ======= 【全新升级】深度挖掘时间维度构建特征 =======
    if 'created_str' in df.columns:
        dt_series = pd.to_datetime(df['created_str'])
        df['created_date'] = dt_series.dt.date
        df['Year-Month'] = dt_series.dt.to_period('M').astype(str) # 转换为 2026-06 格式的字符串便于统计
        df['Year-Month'] = dt_series.dt.strftime('%Y-%m')
        df['Hour'] = dt_series.dt.hour # 提取小时 0-23
        df['Day_of_Week'] = dt_series.dt.day_name() # 提取周几 (Monday, Tuesday...)
        
        # 方便排序周几
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=weekday_order, ordered=True)
        
    return df

try:
    with st.spinner("🤖 大模型正在火力全开，提取时间序列特征并进行舆情分析..."):
        df_raw = load_and_analyze_data()

    # ================= 5. 侧边栏 =================
    st.sidebar.title("🎛️ 社区过滤器 (Filters)")
    sentiment_filter = st.sidebar.multiselect(
        "选择你想观察的用户情绪", 
        options=["negative", "neutral", "positive"], 
        default=["negative", "neutral", "positive"]
    )
    min_score = st.sidebar.slider("最少获得的点赞互动数 (Score)", 0, int(df_raw['score'].max()), 0)
    search_query = st.sidebar.text_input("🔍 输入关键词检索帖子内容")

    # 动态过滤
    df = df_raw[df_raw['sentiment'].isin(sentiment_filter)]
    df = df[df['score'] >= min_score]
    if search_query:
        df = df[df['title'].str.contains(search_query, case=False) | df['selftext'].str.contains(search_query, case=False)]

    # ================= 6. 主视窗 =================
    st.title("🕶️ Rokid Community AI Analytics Insight Pro")
    st.caption("融合大语言模型 (LLM) 与多维时间周期性矩阵的海外极客生态看板")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("📦 当前筛选帖数", len(df))
    with m2: st.metric("🔥 互动总点赞数", int(df['score'].sum()))
    with m3: st.metric("💬 累计评论互动", int(df['num_comments'].sum()))
    with m4:
        neg_rate = f"{round((df_raw['sentiment']=='negative').sum() / len(df_raw) * 100, 1)}%"
        st.metric("🚨 社区整体负面率", neg_rate)

    st.markdown("---")

    # 重新编排的四大导航页
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 TIME TREND (月/周/日多维趋势)", 
        "🔤 LANGUAGE (核心词汇下钻)", 
        "💙 AI SENTIMENT (情绪健康度分析)",
        "📋 DATA WALL (原始反馈精选墙)"
    ])

    # ================= TAB 1: 【全新重写】月/周/日多维时间趋势 =================
    with tab1:
        st.subheader("⏱️ 宏观与微观时间序列周期性分析 (Time-series & Seasonality)")
        
        # 1. 第一排：宏观历史趋势（月度 vs 每日）
        c_macro_left, c_macro_right = st.columns(2)
        with c_macro_left:
            st.markdown("##### 📅 月度大盘推移 (Monthly Trend)")
            df_month = df.groupby('Year-Month').size().reset_index(name='Post Count')
            fig_month = px.bar(df_month, x='Year-Month', y='Post Count', template="plotly_dark")
            fig_month.update_traces(marker_color='#a64dff')
            st.plotly_chart(fig_month, use_container_width=True)
            
        with c_macro_right:
            st.markdown("##### 📆 每日连续波动 (Daily Trend)")
            df_time = df.groupby('created_date').size().reset_index(name='Count')
            fig_daily = px.line(df_time, x='created_date', y='Count', template="plotly_dark")
            fig_daily.update_traces(line_color='#00ffcc', line_width=2)
            st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("---")
        
        # 2. 第二排：微观周期规律（周几活跃度 vs 24小时发帖习惯）
        c_micro_left, c_micro_right = st.columns(2)
        with c_micro_left:
            st.markdown("##### 🗓️ 星期活跃度周期规律 (Weekly Seasonality - Which day is hot?)")
            df_week = df.groupby('Day_of_Week').size().reset_index(name='Post Count')
            fig_week = px.bar(df_week, x='Day_of_Week', y='Post Count', template="plotly_dark")
            fig_week.update_traces(marker_color='#ff3399')
            st.plotly_chart(fig_week, use_container_width=True)
            
        with c_micro_right:
            st.markdown("##### ⏰ 24小时用户在线发帖时间盘 (Daily Hourly Distribution - UTC/Server Time)")
            df_hour = df.groupby('Hour').size().reset_index(name='Post Count')
            # 使用折线或平滑面积图展现一天之中的用户习惯
            fig_hour = px.area(df_hour, x='Hour', y='Post Count', template="plotly_dark")
            fig_hour.update_traces(line_color='#ffcc00', fillcolor='rgba(255, 204, 0, 0.2)')
            fig_hour.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2)) # 强制2小时显示一格
            st.plotly_chart(fig_hour, use_container_width=True)

    # ================= TAB 2: LANGUAGE =================
    with tab2:
        st.subheader("🕵️‍♂️ 关键词深度下钻与热力图")
        all_text = " ".join(df['title'].dropna().tolist() + df['selftext'].dropna().tolist()).lower()
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        filtered_words = [w for w in words if w not in STOPWORDS]
        word_counts = collections.Counter(filtered_words)
        df_words = pd.DataFrame(word_counts.most_common(25), columns=['Word', 'Count'])
        
        l_chart, r_table = st.columns([2, 1])
        with l_chart:
            fig_words = px.bar(df_words, x='Count', y='Word', orientation='h', template="plotly_dark")
            fig_words.update_traces(marker_color='#ffaa00')
            fig_words.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
        with r_table:
            st.dataframe(df_words, use_container_width=True, hide_index=True)

    # ================= TAB 3: SENTIMENT =================
    with tab3:
        st.subheader("🧠 Hugging Face 深度情绪穿透")
        c1, c2 = st.columns(2)
        color_map = {"negative": "#ff4b4b", "positive": "#00cc96", "neutral": "#636efa"}
        with c1:
            sentiment_counts = df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', color='Sentiment', color_discrete_map=color_map, template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.markdown("##### 📊 情绪时间交叉堆叠图")
            df_sent_time = df.groupby(['created_date', 'sentiment']).size().reset_index(name='Count')
            fig_sent_line = px.bar(df_sent_time, x='created_date', y='Count', color='sentiment', color_discrete_map=color_map, template="plotly_dark", barmode='stack')
            st.plotly_chart(fig_sent_line, use_container_width=True)

    # ================= TAB 4: DATA WALL =================
    with tab4:
        st.subheader("📌 用户反馈内容精选墙 (User Feedback Wall)")
        for idx, row in df.head(20).iterrows():
            color = "🔴 [负面]" if row['sentiment'] == 'negative' else ("🟢 [正面]" if row['sentiment'] == 'positive' else "🔵 [中性]")
            with st.expander(f"{color} 赞同数: {row['score']} | {row['title']}"):
                st.markdown(f"**作者：** `{row['author_name']}`  |  **日期：** {row['created_str']}  |  **星期：** {row.get('Day_of_Week', '未知')}")
                st.info(row['selftext'] if row['selftext'] else "（该贴无正文，仅有标题）")

except Exception as e:
    st.error(f"看板更新失败，错误信息: {e}")