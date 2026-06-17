import streamlit as st
import pandas as pd
import plotly.express as px
import json
import collections
import re
from transformers import pipeline

# 1. 页面基本配置
st.set_page_config(layout="wide", page_title="Rokid Reddit AI Analytics")

# 英文停用词表
STOPWORDS = set(["the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "for", "in", "on", "at", "by", "with", "of", "it", "its", "this", "that", "i", "you", "he", "she", "they", "we", "my", "your", "have", "has", "can", "will", "do", "not", "just", "so", "if", "about", "like", "any", "get", "me", "from", "up", "out", "how", "what", "when", "some", "more", "there", "would", "am", "or", "out"] + [str(i) for i in range(10)])

# 2. 初始化 Hugging Face 情感分析流水线 (加缓存防止重复下载)
@st.cache_resource
def load_sentiment_pipeline():
    # 选用轻量且对网络网络语境敏感的 RoBERTa 模型
    return pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

# 3. 读取并清洗数据（包含 AI 情感打标）
@st.cache_data
def load_and_analyze_data():
    file_path = "rokid_official_full_data2.json"
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    sentiment_task = load_sentiment_pipeline()
    flattened_list = []
    
    # 为了防止本地跑模型太慢，这里先默认分析前 200 条，你想全跑可以把 [:200] 删掉
    for item in raw_data:
        meta = item.get("meta", {})
        content = item.get("content", {})
        stats = item.get("stats", {})
        author = item.get("author", {})
        
        title = content.get("title", "")
        selftext = content.get("selftext", "")
        full_text = f"{title}. {selftext}"[:512] # 截断防止超出模型长度上限
        
        # 默认中性
        sentiment_label = "neutral"
        if full_text.strip():
            try:
                # 让大模型去读文本
                res = sentiment_task(full_text)[0]
                sentiment_label = res['label'] # 得到 positive / negative / neutral
            except:
                pass

        row = {
            "id": meta.get("id"),
            "created_str": meta.get("created_str"),
            "title": title,
            "selftext": selftext,
            "score": stats.get("score", 0),
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
    # 加载带有 AI 标签的数据
    with st.spinner("🤖 Hugging Face 大模型正在疯狂计算、对社区文本进行情感分类...请稍候..."):
        df = load_and_analyze_data()

    # 统一头部
    st.title("🕶️ Rokid Community AI Analytics Report")
    st.caption("基于大模型 (LLM/NLP) 与传统统计学的 Reddit 深度挖掘看板")
    st.markdown("---")

    # 三个核心标签页
    tab1, tab2, tab3 = st.tabs(["💬 MESSAGES (基础统计)", "🔤 LANGUAGE (核心词汇)", "💙 AI SENTIMENT (情绪与痛点)"])

    # ================= TAB 1 & 2 保持原样 =================
    with tab1:
        left_col, right_col = st.columns([3, 1])
        with left_col:
            st.subheader("Messages sent over time by day")
            df_time = df.groupby('created_date').size().reset_index(name='Message Count')
            fig = px.line(df_time, x='created_date', y='Message Count', template="plotly_dark")
            fig.update_traces(line_color='#00ffcc')
            st.plotly_chart(fig, use_container_width=True)
        with right_col:
            st.subheader("📊 Statistics")
            st.metric("Analyzed Posts (已分析帖数)", len(df))
            st.metric("Total Engagement (总点赞数)", int(df['score'].sum()))
            st.metric("Total Comments (总评论数)", int(df['num_comments'].sum()))

    with tab2:
        st.subheader("🔤 Word Frequency (用户高频提及词汇)")
        all_text = " ".join(df['title'].dropna().tolist() + df['selftext'].dropna().tolist()).lower()
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        filtered_words = [w for w in words if w not in STOPWORDS]
        word_counts = collections.Counter(filtered_words)
        df_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
        
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            fig_words = px.bar(df_words, x='Count', y='Word', orientation='h', template="plotly_dark")
            fig_words.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
        with col_table:
            st.dataframe(df_words, use_container_width=True)

    # ================= TAB 3: AI SENTIMENT (全新解锁) =================
    with tab3:
        st.subheader("🤖 Hugging Face 情感分类矩阵")
        
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            # 统计各种情绪的占比
            sentiment_counts = df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            # 统一颜色映射：负面用红色，正面用绿色，中性用灰色
            color_map = {"negative": "#ff4b4b", "positive": "#00cc96", "neutral": "#636efa"}
            
            fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                             title="社区舆情健康度占比 (Sentiment Share)",
                             color='Sentiment', color_discrete_map=color_map,
                             template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_bar:
            # 联动分析：负面情绪在哪些时间点爆发？
            st.write("📊 舆情随时间的演变趋势")
            df_sent_time = df.groupby(['created_date', 'sentiment']).size().reset_index(name='Count')
            fig_sent_line = px.bar(df_sent_time, x='created_date', y='Count', color='sentiment',
                                   color_discrete_map=color_map, template="plotly_dark", barmode='stack')
            st.plotly_chart(fig_sent_line, use_container_width=True)
            
        st.markdown("---")
        # 负面用户声音提取窗口（产品优化最核心的参考依据）
        st.subheader("🚨 重点关注：海外用户负面反馈/Bug 实时看板")
        
        df_negative = df[df['sentiment'] == 'negative'][['created_date', 'author_name', 'title', 'selftext', 'score']]
        if not df_negative.empty:
            st.warning(f"💡 AI 为您过滤出了 {len(df_negative)} 条负面反馈帖子。建议产品和研发团队重点排查：")
            st.dataframe(df_negative.sort_values(by='score', ascending=False), use_container_width=True)
        else:
            st.success("🎉 太棒了，当前分析的数据集中未发现明显的负面情绪帖子！")

except Exception as e:
    st.error(f"AI 看板构建失败，错误信息: {e}")
    