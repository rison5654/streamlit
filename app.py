import streamlit as st
import pandas as pd
import requests
import datetime
from datetime import date
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

# --- 1. 기본 페이지 설정 ---
st.set_page_config(
    page_title="일본 여행지 추천 플래너 (v22)",
    page_icon="🇯🇵",
    layout="wide",
)

# --- Gist API URL ---
# !!! v19/v20/v21용 Gist URL (일본 7개 도시)을 붙여넣으세요 !!!
TRAVEL_DATA_URL = "https://gist.githubusercontent.com/rison5654/a05581379502b5cfc773a286aea40aa9/raw/fa8fd366fead395457cdf65495632feb8ae569e7/travel_db.json" 

# --- (신규) 2. 색상 정의 ---
# ML 클러스터용 색상
ML_CLUSTER_COLORS = ["#FF4B4B", "#0068C9", "#0ABF53", "#FFA400", "#800080", "#A52A2A"]
# 날씨 맵용 색상
WEATHER_COLOR_MAP = {
    "☀️": "#FFA500", # 맑음 (주황)
    "🌤️": "#FFC300", # 대체로 맑음
    "☁️": "#B0B0B0", # 흐림 (회색)
    "🌫️": "#D3D3D3", # 안개
    "🌦️": "#70A0FF", # 이슬비 (파랑)
    "🌧️": "#0068C9", # 비 (진한 파랑)
    "❄️": "#FFFFFF", # 눈 (흰색)
    "🌨️": "#E0E0E0", # 눈보라
    "⛈️": "#4B0082", # 뇌우 (남색)
    "❓": "#303030"  # 오류 (검정)
}


# --- 3. 데이터 로드 (Gist API 호출) ---
@st.cache_data(ttl=3600)
def load_travel_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df["Avg_Cost_KRW"] = pd.to_numeric(df["Avg_Cost_KRW"])
        return df
    except requests.RequestException as e:
        st.error(f"여행지 데이터 API 호출 오류: {e}")
        return pd.DataFrame()

# --- 4. K-Means 엘보우 메소드 계산 함수 (v20과 동일) ---
@st.cache_data
def calculate_elbow_data(df_features):
    if len(df_features) < 2: return None
    features_scaled = StandardScaler().fit_transform(df_features)
    inertia_list = []
    max_k = min(len(features_scaled) - 1, 6) 
    if max_k < 2: return None
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertia_list.append(kmeans.inertia_)
    elbow_df = pd.DataFrame({'K (클러스터 수)': k_range, 'Inertia (응집도)': inertia_list})
    return elbow_df

# --- 5. 추천 점수(Score) 계산 함수 (v20과 동일) ---
def calculate_recommendation_score(df, selected_types):
    if df.empty:
        df['Score'] = 0
        return df
    scaler = MinMaxScaler()
    df['Budget_Score'] = scaler.fit_transform(-df['Avg_Cost_KRW'].values.reshape(-1, 1)) * 100
    def type_match_score(row_types):
        if not selected_types: 
            return 0
        db_type_list = [t.strip() for t in row_types.split(',')]
        match_count = sum(1 for t in selected_types if t in db_type_list)
        return (match_count / len(selected_types)) * 100 
    df['Type_Score'] = df['Type'].apply(type_match_score)
    df['Score'] = (df['Budget_Score'] * 0.5) + (df['Type_Score'] * 0.5)
    df['Score'] = df['Score'].round(0).astype(int)
    return df

# --- 6. "7일 예보" API 호출 함수 (v21과 동일) ---
@st.cache_data(ttl=600)
def get_weather_forecast(latitude, longitude):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude, "longitude": longitude,
            "daily": "weathercode,temperature_2m_max,temperature_2m_min",
            "timezone": "auto", "forecast_days": 7
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()["daily"]
    except requests.RequestException:
        return None

# --- 7. 날씨 코드 -> 이모지 변환 함수 (v21과 동일) ---
def weather_code_to_emoji(code):
    if code is None: return "❓"
    if code == 0: return "☀️"
    if code in [1, 2, 3]: return "🌤️"
    if code in [45, 48]: return "🌫️"
    if code in [51, 53, 55, 56, 57]: return "🌦️"
    if code in [61, 63, 65, 66, 67]: return "🌧️"
    if code in [71, 73, 75, 77]: return "❄️"
    if code in [80, 81, 82]: return "🌦️"
    if code in [85, 86]: return "🌨️"
    if code in [95, 96, 99]: return "⛈️"
    return "☁️"

# --- (신규) 8. "오늘 날씨" 융합 및 색상 매핑 (v10 로직 부활) ---
@st.cache_data(ttl=600) # 10분 캐시
def get_and_merge_today_weather(df):
    """Gist의 모든 도시에 대해 API를 호출하고 '오늘' 날씨 정보를 병합합니다."""
    today_emojis = []
    today_colors = []
    
    for _, row in df.iterrows():
        # 개별 API 호출은 캐시된 함수를 사용
        forecast = get_weather_forecast(row['Latitude'], row['Longitude'])
        
        if forecast:
            today_code = forecast['weathercode'][0]
            today_emoji = weather_code_to_emoji(today_code)
            today_emojis.append(today_emoji)
            # 이모지를 기반으로 맵용 색상 매핑
            today_colors.append(WEATHER_COLOR_MAP.get(today_emoji, "#303030"))
        else:
            today_emojis.append("❓")
            today_colors.append(WEATHER_COLOR_MAP["❓"])
    
    df['Today_Weather_Emoji'] = today_emojis
    df['Today_Weather_Color'] = today_colors # <-- 날씨 맵을 위한 새 컬럼
    return df

# --- 데이터 로딩 실행 ---
with st.spinner("일본 여행지 목록 API 로딩 중... (Gist)"):
    df_travel_base = load_travel_data(TRAVEL_DATA_URL)

# (신규) Gist 로딩 직후, "오늘 날씨" 정보도 융합
if not df_travel_base.empty:
    with st.spinner("모든 도시의 '오늘' 실시간 날씨 로딩 중... (Meteo API)"):
        df_travel_base = get_and_merge_today_weather(df_travel_base)

# --- 9. 사이드바 (필터) (v20과 동일) ---
st.sidebar.header("🇯🇵 일본 여행 플래너")
if df_travel_base.empty:
    st.sidebar.error("데이터 로딩 실패. Gist URL을 확인하세요.")
    n_clusters = 3 
    selected_budget_range = (0, 0)
    selected_types = []
    selected_season = "전체"
else:
    all_types = set()
    df_travel_base['Type'].str.split(',').apply(lambda x: [all_types.add(t.strip()) for t in x])
    sorted_types = sorted(list(all_types))
    selected_types = st.sidebar.multiselect("1. 원하는 여행 타입은? (다중 선택)", sorted_types, default=["관광", "미식"])
    all_seasons = ["전체", "봄", "여름", "가을", "겨울"]
    selected_season = st.sidebar.radio("2. 여행할 계절은?", all_seasons, horizontal=True)
    min_budget = int(df_travel_base["Avg_Cost_KRW"].min())
    max_budget = int(df_travel_base["Avg_Cost_KRW"].max())
    selected_budget_range = st.sidebar.slider(f"3. 1일 예산 범위 (KRW 원)", min_value=min_budget, max_value=max_budget, value=(min_budget, max_budget), step=10000 )
    st.sidebar.divider()
    st.sidebar.header("🤖 ML 분석 설정")
    n_clusters = st.sidebar.number_input("유사 여행지 그룹 수 (K)", min_value=2, max_value=5, value=3, step=1, help="전체 도시의 '예산, 위도, 경도'를 기준으로 군집화합니다.")

# --- 10. ML 클러스터링 (전체 데이터 대상) (v20과 동일) ---
if not df_travel_base.empty:
    features_for_clustering = df_travel_base[['Avg_Cost_KRW', 'Latitude', 'Longitude']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_clustering)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    clusters_int = kmeans.fit_predict(features_scaled)
    df_travel_base['Cluster'] = clusters_int.astype(str)
    # (변경) ML 맵을 위한 색상 컬럼 (v6 로직)
    df_travel_base['ML_Map_Color'] = [ML_CLUSTER_COLORS[i] for i in clusters_int]

# --- 11. 데이터 필터링 및 스코어링 (v20과 동일) ---
if not df_travel_base.empty:
    filtered_df = df_travel_base.copy()
    if selected_season != "전체":
        filtered_df = filtered_df[filtered_df['Best_Season'].str.contains(selected_season)]
    if selected_types:
        or_condition = '|'.join(selected_types)
        filtered_df = filtered_df[filtered_df['Type'].str.contains(or_condition)]
    filtered_df = filtered_df[
        (filtered_df["Avg_Cost_KRW"] >= selected_budget_range[0]) &
        (filtered_df["Avg_Cost_KRW"] <= selected_budget_range[1])
    ]
    filtered_df = calculate_recommendation_score(filtered_df, selected_types)
    filtered_df = filtered_df.sort_values(by="Score", ascending=False)
else:
    filtered_df = pd.DataFrame()

# --- 12. 메인 페이지 (시각화) ---
st.title("🇯🇵 일본 여행지 추천 플래너 (v22)")
st.markdown(f"**선택 조건:** `예산( {selected_budget_range[0]:,}원 ~ {selected_budget_range[1]:,}원 )`, `타입( {', '.join(selected_types)} )`, `계절( {selected_season} )`")

# 12-1. 카드 UI (v20과 동일, 'Today_Weather_Emoji' 컬럼 활용)
st.divider()
st.subheader(f"🏆 {selected_season} 여행을 위한 BEST 추천 (Score 기반)")
if filtered_df.empty:
    st.warning("조건에 맞는 여행지가 없습니다. 필터를 조정해 주세요.")
else:
    top5_df = filtered_df.head(5)
    for _, row in top5_df.iterrows():
        with st.container(border=True):
            col_score, col_info, col_cost, col_weather = st.columns([1, 3, 2, 2])
            with col_score:
                st.metric(label="추천 점수", value=f"{row['Score']}/100")
            with col_info:
                st.subheader(f"📍 {row['City']}, {row['Country']}")
                st.caption(f"타입: {row['Type']} | {row['Description']}")
            with col_cost:
                st.metric(label="1일 예산", value=f"{row['Avg_Cost_KRW']:,} 원")
            with col_weather:
                # (신규) '오늘 날씨' 표시
                st.metric(label="오늘의 날씨", value=f"{row['Today_Weather_Emoji']}")
        st.write("") 

# 12-2. 기술 분석 탭 (3개 탭) (v21과 동일)
st.divider()
tab1, tab2, tab3 = st.tabs([
    "🤖 전체 도시 기술 분석 (ML)",
    "📋 필터링된 상세 데이터",
    "🌦️ 실시간 날씨 (API)" # 이름 변경
])

with tab1:
    # (v20과 동일, 단 color='ML_Map_Color'로 변경)
    st.subheader("전체 도시 분포 (ML 클러스터링 적용)")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("**ML 클러스터(경제/지리) 지도**")
        if df_travel_base.empty: st.warning("Gist 데이터를 로드하지 못했습니다.")
        else: st.map(df_travel_base, latitude='Latitude', longitude='Longitude', color='ML_Map_Color') # <-- ML 맵
    with col2:
        st.markdown("**예산별 여행지 분포 (Bar Chart)**")
        if not df_travel_base.empty and len(df_travel_base) > 0:
            # (v20 코드와 동일)
            min_data = int(df_travel_base['Avg_Cost_KRW'].min()); min_val = (min_data // 100000) * 100000 
            max_val = int(df_travel_base['Avg_Cost_KRW'].max()); max_bin = (max_val // 100000 + 1) * 100000 
            if max_bin == 0: max_bin = 100000 
            bin_edges = list(range(min_val, max_bin + 100000, 100000))
            if not bin_edges: bin_edges = [min_val, max_bin]
            bin_labels = [];
            for i in range(len(bin_edges) - 1):
                start_label = f"{bin_edges[i]//10000}만"; end_label = f"{bin_edges[i+1]//10000}만"
                if i == len(bin_edges) - 2: bin_labels.append(f"{start_label} 이상")
                else: bin_labels.append(f"{start_label}~{end_label}")
            if not bin_labels: bin_labels = [f"{min_val}~{max_bin}"] 
            bins = pd.cut(df_travel_base['Avg_Cost_KRW'], bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)
            df_travel_base['Budget_Bin'] = bins.astype(str)
            plot_df = df_travel_base.groupby(['Budget_Bin', 'Cluster']).agg(Count=('City', 'size'), Cities=('City', lambda x: ', '.join(x)), Countries=('Country', lambda x: ', '.join(x.unique()))).reset_index()
            fig = px.bar(plot_df, x='Budget_Bin', y='Count', color='Cluster', color_discrete_map={str(i): color for i, color in enumerate(ML_CLUSTER_COLORS)}, hover_data=['Cities', 'Countries'], title="선택된 여행지의 예산 분포")
            fig.update_layout(xaxis_title="1. 평균 예산 (KRW)", yaxis_title="도시 수")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("표시할 차트 데이터가 없습니다.")
            
    st.divider()
    with st.expander("ℹ️ (기술 증명) K-Means 엘보우 메소드(Elbow Method)"):
        if not df_travel_base.empty:
            st.markdown("그래프에서 '팔꿈치'처럼 꺾이는 지점이 가장 효율적인 K값입니다.")
            elbow_features = df_travel_base[['Avg_Cost_KRW', 'Latitude', 'Longitude']]
            elbow_df = calculate_elbow_data(elbow_features)
            if elbow_df is not None:
                fig_elbow = px.line(elbow_df, x='K (클러스터 수)', y='Inertia (응집도)', title="K값에 따른 Inertia 변화 (엘보우 메소드)", markers=True)
                fig_elbow.update_traces(marker=dict(size=8))
                st.plotly_chart(fig_elbow, use_container_width=True)
            else: st.warning("데이터가 너무 적어 엘보우 차트를 생성할 수 없습니다.")
        else: st.info("먼저 Gist 데이터를 로드해주세요.")

with tab2:
    # (v20과 동일)
    st.subheader("📋 필터링된 여행지 상세 데이터 (Score 기준 정렬)")
    st.markdown("사이드바의 필터 조건에 따라 실시간으로 필터링되며, 추천 점수(Score)가 높은 순으로 정렬됩니다.")
    if not filtered_df.empty:
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    else:
        st.warning("조건에 맞는 여행지가 없습니다.")

# --- (신규) 10-3. 실시간 날씨 탭 (v21 수정) ---
with tab3:
    st.subheader("🌦️ 7일간 실시간 날씨 (Live API)")
    st.markdown("출발 1주일 전, 이 탭에서 도시의 실시간 날씨를 확인할 수 있습니다.")
    
    if df_travel_base.empty:
        st.warning("Gist 데이터를 로드하지 못했습니다.")
    else:
        # --- (신규) 날씨 맵 ---
        st.markdown("**일본 전역 '오늘' 날씨 지도**")
        st.map(df_travel_base, 
               latitude='Latitude', 
               longitude='Longitude', 
               color='Today_Weather_Color') # <-- 날씨 맵
        st.caption(f"☀️(맑음) 🌧️(비) ❄️(눈) ☁️(흐림)")
        st.divider()
        # ---
        
        # 7일 상세 예보 (v21과 동일)
        all_city_list = sorted(df_travel_base['City'].tolist())
        selected_city_for_weather = st.selectbox(
            "7일간 상세 예보를 볼 도시를 선택하세요:", 
            all_city_list, 
            key="weather_city_select"
        )
        
        if selected_city_for_weather:
            city_info = df_travel_base[df_travel_base['City'] == selected_city_for_weather].iloc[0]
            lat = city_info['Latitude']
            lon = city_info['Longitude']
            
            st.markdown(f"**{selected_city_for_weather}** (위도: {lat}, 경도: {lon})의 7일 예보입니다.")
            
            with st.spinner(f"{selected_city_for_weather}의 실시간 날씨 API 로딩 중..."):
                forecast_data = get_weather_forecast(lat, lon)
            
            if forecast_data:
                cols = st.columns(7)
                today = date.today()
                
                for i in range(7):
                    with cols[i]:
                        day = today + datetime.timedelta(days=i)
                        st.markdown(f"**{day.strftime('%m/%d')}**") 
                        
                        emoji = weather_code_to_emoji(forecast_data["weathercode"][i])
                        st.markdown(f"<h1 style='text-align: center; margin: 0;'>{emoji}</h1>", unsafe_allow_html=True)
                        
                        st.metric(
                            label="최고/최저",
                            value=f"{forecast_data['temperature_2m_max'][i]}°C",
                            delta=f"{forecast_data['temperature_2m_min'][i]}°C",
                            delta_color="off"
                        )
            else:
                st.error("날씨 정보를 불러오는 데 실패했습니다.")