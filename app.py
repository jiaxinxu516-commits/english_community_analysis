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

# 2. 页面全局配置（宽屏暗黑风）
st.set_page_config(layout="wide", page_title="Rokid Reddit AI Dashboard")

# 英文停用词表
STOPWORDS = set(["the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "for", "in", "on", "at", "by", "with", "of", "it", "its", "this", "that", "i", "you", "he", "she", "they", "we", "my", "your", "have", "has", "can", "will", "do", "not", "just", "so", "if", "about", "like", "any", "get", "me", "from", "up", "out", "how", "what", "when", "some", "more", "there", "would", "am", "or", "out", "want", "one", "know", "has", "been", "me", "dont", "think", "anyone", "even", "still"] + [str(i) for i in range(10)])

# 3. 初始化大模型（加资源缓存）
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

# 4. 读取全量数据并用 AI 批量打标
@st.cache_data
def load_and_analyze_data():
    file_path = "rokid_official_full_data2.json"
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    sentiment_task = load_sentiment_pipeline()
    flattened_list = []
    
    # 【已解锁】这里去掉了 [:200] 的限制，直接跑全量数据。
    # 为了防止全量太庞大导致页面初次加载过慢，我们这里限制前 600 条（基本覆盖了核心内容）
    # 如果你的电脑性能好，想跑完 6.6MB 的全部几千条，可以将 [:600] 删掉。
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
    if 'created_str' in df.columns:
        df['created_date'] = pd.to_datetime(df['created_str']).dt.date
    return df

try:
    # 加载 AI 分析完毕的数据
    with st.spinner("🤖 大模型正在火力全开，全量对 Reddit 社区文本进行深度舆情打标..."):
        df_raw = load_and_analyze_data()

    # ================= 5. 侧边栏交互联动（像原图一样可以筛选） =================
    st.sidebar.title("🎛️ 社区过滤器 (Filters)")
    
    # 筛选1：情感类型筛选
    sentiment_filter = st.sidebar.multiselect(
        "选择你想观察的用户情绪", 
        options=["negative", "neutral", "positive"], 
        default=["negative", "neutral", "positive"]
    )
    
    # 筛选2：互动热度筛选
    min_score = st.sidebar.slider("最少获得的点赞互动数 (Score)", 0, int(df_raw['score'].max()), 0)
    
    # 筛选3：关键词搜索栏
    search_query = st.sidebar.text_input("🔍 输入关键词检索帖子内容 (如: station, bug, battery)")

    # 根据侧边栏的选择动态过滤数据
    df = df_raw[df_raw['sentiment'].isin(sentiment_filter)]
    df = df[df['score'] >= min_score]
    if search_query:
        df = df[df['title'].str.contains(search_query, case=False) | df['selftext'].str.contains(search_query, case=False)]

    # ================= 6. 主视窗内容填充 =================
    st.title("🕶️ Rokid Community Analytics Insight Pro")
    st.caption("融合大语言模型 (LLM) 与社区互动矩阵的海外极客生态看板")
    
    # 顶部关键核心指标栏 (大数字卡片)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("📦 当前筛选帖数", len(df))
    with m2:
        st.metric("🔥 互动总点赞数", int(df['score'].sum()))
    with m3:
        st.metric("💬 累计评论互动", int(df['num_comments'].sum()))
    with m4:
        neg_rate = f"{round((df_raw['sentiment']=='negative').sum() / len(df_raw) * 100, 1)}%"
        st.metric("🚨 社区整体负面率", neg_rate)

    st.markdown("---")

    # 选项卡切换
    tab1, tab2, tab3 = st.tabs(["💬 MESSAGES (趋势与活跃)", "🔤 LANGUAGE (词频多维挖掘)", "💙 AI SENTIMENT (情绪深度洞察)"])

    # --- TAB 1 ---
    with tab1:
        left, right = st.columns([3, 1])
        with left:
            st.subheader("📊 每日社区内容发布与活跃趋势 (Daily Message Trend)")
            df_time = df.groupby('created_date').size().reset_index(name='Count')
            fig = px.bar(df_time, x='created_date', y='Count', template="plotly_dark")
            fig.update_traces(marker_color='#00ffcc')
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("🏆 社区核心贡献者排行")
            top_authors = df['author_name'].value_counts().head(8).reset_index()
            top_authors.columns = ['用户名', '发帖频次']
            st.dataframe(top_authors, use_container_width=True, hide_index=True)

    # --- TAB 2 ---
    with tab2:
        st.subheader("🕵️‍♂️ 关键词深度下钻与热力图")
        
        # 动态提取当前过滤条件下文本的词频
        all_text = " ".join(df['title'].dropna().tolist() + df['selftext'].dropna().tolist()).lower()
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        filtered_words = [w for w in words if w not in STOPWORDS]
        word_counts = collections.Counter(filtered_words)
        df_words = pd.DataFrame(word_counts.most_common(25), columns=['Word', 'Count'])
        
        l_chart, r_table = st.columns([2, 1])
        with l_chart:
            fig_words = px.bar(df_words, x='Count', y='Word', orientation='h', 
                               template="plotly_dark", title="用户最关心的话题/硬件部件 Top 25")
            fig_words.update_traces(marker_color='#ffaa00')
            fig_words.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
        with r_table:
            st.markdown("**高频词密度明细**")
            st.dataframe(df_words, use_container_width=True, hide_index=True)

    # --- TAB 3 ---
    with tab3:
        st.subheader("🧠 Hugging Face 深度情绪穿透矩阵")
        
        c1, c2 = st.columns(2)
        color_map = {"negative": "#ff4b4b", "positive": "#00cc96", "neutral": "#636efa"}
        
        with c1:
            sentiment_counts = df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                             title="当前筛选条件下的舆情占比",
                             color='Sentiment', color_discrete_map=color_map, template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.markdown("##### 📌 用户反馈内容精选墙 (User Feedback Wall)")
            st.write("点击侧边栏过滤条件，下方看板内容将**实时动态洗牌切换**：")
            
            # 用优雅的形式把帖子循环打印出来，不再是光秃秃的表格
            for idx, row in df.head(15).iterrows():
                # 根据不同情绪显示不同的色彩边框
                color = "🔴 [负面]" if row['sentiment'] == 'negative' else ("🟢 [正面]" if row['sentiment'] == 'positive' else "🔵 [中性]")
                with st.expander(f"{color} 赞同数: {row['score']} | {row['title']}"):
                    st.markdown(f"**作者：** `{row['author_name']}`  |  **日期：** {row['created_date']}")
                    st.info(row['selftext'] if row['selftext'] else "（该贴无正文，仅有标题）")

except Exception as e:
    st.error(f"看板更新失败，错误信息: {e}")