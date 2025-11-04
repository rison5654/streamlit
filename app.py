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
    page_title="일본 여행지 추천 플래너",
    page_icon="🇯🇵",
    layout="wide",
)

# --- Gist API URL ---
# !!! v26용 Gist URL (일본 20개 도시)을 붙여넣으세요 !!!
TRAVEL_DATA_URL = "https://gist.githubusercontent.com/rison5654/a05581379502b5cfc773a286aea40aa9/raw/6c77281b7613f8fe99076d60b1efeb901990f697/travel_db.json" 

# --- 2. 색상 정의 ---
ML_CLUSTER_COLORS = [
    "#FF4B4B", "#0068C9", "#0ABF53", "#FFA400", 
    "#800080", "#A52A2A", "#FFC0CB", "#00FFFF" # 8개
]
WEATHER_COLOR_MAP = {
    "☀️": "#FFA500", "🌤️": "#FFC300", "☁️": "#B0B0B0", "🌫️": "#D3D3D3",
    "🌦️": "#70A0FF", "🌧️": "#0068C9", "❄️": "#FFFFFF", "🌨️": "#E0E0E0",
    "⛈️": "#4B0082", "❓": "#303030"
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

# --- 4. K-Means 엘보우 메소드 계산 함수 ---
@st.cache_data
def calculate_elbow_data(df_features):
    if len(df_features) < 2: return None
    features_scaled = StandardScaler().fit_transform(df_features)
    inertia_list = []
    max_k = min(len(features_scaled) - 1, 10) # 20개 도시
    if max_k < 2: return None
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertia_list.append(kmeans.inertia_)
    elbow_df = pd.DataFrame({'K (클러스터 수)': k_range, 'Inertia (응집도)': inertia_list})
    return elbow_df

# --- 실시간 환율 API 호출 ---
@st.cache_data(ttl=3600) # 1시간 캐시
def get_jpy_rate():
    try:
        response = requests.get("https://api.frankfurter.app/latest?from=KRW&to=JPY")
        return response.json()['rates']['JPY']
    except Exception:
        return 0.11 # API 실패 시 기본 환율

jpy_rate = get_jpy_rate()

# --- (변경) 5. 추천 점수(Score) 계산 함수 ---
def calculate_recommendation_score(df, selected_types, weight_budget_percent):
    """
    필터링된 DataFrame을 받아 'Score' 컬럼을 추가합니다.
    (가중치를 파라미터로 받음)
    """
    if df.empty:
        df['Score'] = 0
        return df

    scaler = MinMaxScaler()

    # 1. 예산 점수 (v26과 동일)
    df['Budget_Score'] = scaler.fit_transform(-df['Avg_Cost_KRW'].values.reshape(-1, 1)) * 100

    # 2. 여행 타입 일치도 점수 (v26과 동일)
    def type_match_score(row_types):
        if not selected_types: 
            return 0
        db_type_list = [t.strip() for t in row_types.split(',')]
        match_count = sum(1 for t in selected_types if t in db_type_list)
        return (match_count / len(selected_types)) * 100 
    df['Type_Score'] = df['Type'].apply(type_match_score)

    # 3. (변경) 최종 점수 (동적 가중치 적용)
    weight_b = weight_budget_percent / 100.0
    weight_t = (100 - weight_budget_percent) / 100.0
    
    df['Score'] = (df['Budget_Score'] * weight_b) + (df['Type_Score'] * weight_t)
    df['Score'] = df['Score'].round(0).astype(int)
    
    return df

# --- 6. "7일 예보" API 호출 함수 (v26과 동일) ---
@st.cache_data(ttl=600)
def get_weather_forecast(latitude, longitude):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": latitude, "longitude": longitude, "daily": "weathercode,temperature_2m_max,temperature_2m_min", "timezone": "auto", "forecast_days": 7}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()["daily"]
    except requests.RequestException:
        return None

# --- 7. 날씨 코드 -> 이모지 변환 함수 (v26과 동일) ---
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

# --- 8. "오늘 날씨" 융합 및 색상 매핑 (v26과 동일) ---
@st.cache_data(ttl=600)
def get_and_merge_today_weather(df):
    today_emojis = []; today_colors = []
    for _, row in df.iterrows():
        forecast = get_weather_forecast(row['Latitude'], row['Longitude'])
        if forecast:
            today_code = forecast['weathercode'][0]
            today_emoji = weather_code_to_emoji(today_code)
            today_emojis.append(today_emoji)
            today_colors.append(WEATHER_COLOR_MAP.get(today_emoji, "#303030"))
        else:
            today_emojis.append("❓"); today_colors.append(WEATHER_COLOR_MAP["❓"])
    df['Today_Weather_Emoji'] = today_emojis
    df['Today_Weather_Color'] = today_colors
    return df

# --- 데이터 로딩 실행 ---
with st.spinner("일본 여행지 목록 API 로딩 중... (Gist, 20도시)"):
    df_travel_base = load_travel_data(TRAVEL_DATA_URL)
if not df_travel_base.empty:
    with st.spinner("모든 도시의 '오늘' 실시간 날씨 로딩 중... (Meteo API)"):
        df_travel_base = get_and_merge_today_weather(df_travel_base)

# --- 9. 사이드바 (필터) ---
st.sidebar.header("🇯🇵 일본 여행 플래너")
if df_travel_base.empty:
    st.sidebar.error("데이터 로딩 실패. Gist URL을 확인하세요.")
    n_clusters = 4; selected_budget_range = (0, 0); selected_types = []; selected_season = "전체"; weight_budget = 50
else:
    # 9-1. 기본 필터 (v26과 동일)
    all_types = set()
    df_travel_base['Type'].str.split(',').apply(lambda x: [all_types.add(t.strip()) for t in x])
    sorted_types = sorted(list(all_types))
    selected_types = st.sidebar.multiselect("1. 원하는 여행 타입은? (다중 선택)", sorted_types, default=["관광", "미식"])
    all_seasons = ["전체", "봄", "여름", "가을", "겨울"]
    selected_season = st.sidebar.radio("2. 여행할 계절은?", all_seasons, horizontal=True)
    min_budget = int(df_travel_base["Avg_Cost_KRW"].min())
    max_budget = int(df_travel_base["Avg_Cost_KRW"].max())
    selected_budget_range = st.sidebar.slider(f"3. 1일 예산 범위 (KRW 원)", min_value=min_budget, max_value=max_budget, value=(min_budget, max_budget), step=10000 )
    
    # --- (신규) 9-2. 가중치 필터 ---
    st.sidebar.divider()
    st.sidebar.header("🏆 Score 가중치 설정")
    weight_budget = st.sidebar.slider(
        "예산(Budget) 점수 비중 (%)", 
        min_value=0, max_value=100, value=50, step=10, # 50% 기본값
        help="예산이 낮을수록 좋은 '예산 점수'의 비중입니다."
    )
    weight_type = 100 - weight_budget
    st.sidebar.markdown(f"**여행 타입(Type) 점수 비중: `{weight_type}%`**")
    # ---
    
    # 9-3. ML 필터 (v26과 동일)
    st.sidebar.divider()
    st.sidebar.header("🤖 ML 분석 설정")
    n_clusters = st.sidebar.number_input(
        "유사 여행지 그룹 수 (K)", 
        min_value=2, max_value=8, value=4, step=1, # 20개 도시
        help="전체 도시의 '예산, 위도, 경도'를 기준으로 군집화합니다."
    )

# --- 10. ML 클러스터링 (전체 데이터 대상) (v26과 동일) ---
if not df_travel_base.empty:
    features_for_clustering = df_travel_base[['Avg_Cost_KRW', 'Latitude', 'Longitude']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_clustering)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    clusters_int = kmeans.fit_predict(features_scaled)
    df_travel_base['Cluster'] = clusters_int.astype(str)
    df_travel_base['ML_Map_Color'] = [ML_CLUSTER_COLORS[i] for i in clusters_int]

# --- (변경) 11. 데이터 필터링 및 스코어링 ---
if not df_travel_base.empty:
    # 11-1. 필터링 (v26과 동일)
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
    
    # 11-2. (변경) 동적 가중치를 함수에 전달
    filtered_df = calculate_recommendation_score(filtered_df, selected_types, weight_budget)
            
    # 11-3. 정렬 (v26과 동일)
    filtered_df = filtered_df.sort_values(by="Score", ascending=False)
        
else:
    filtered_df = pd.DataFrame()

# --- 12. 메인 페이지 (시각화) ---
st.title("🇯🇵 일본 여행지 추천 플래너 (v27 - 가중치)")
st.markdown(f"**선택 조건:** `예산( {selected_budget_range[0]:,}원 ~ {selected_budget_range[1]:,}원 )`, `타입( {', '.join(selected_types)} )`, `계절( {selected_season} )`")

# 12-1. 카드 UI
st.divider()
# (변경) 부제목에 동적 가중치 표시
st.subheader(f"🏆 {selected_season} 여행을 위한 BEST 추천 (Score: 예산 {weight_budget}% : 타입 {100-weight_budget}%)")
if filtered_df.empty:
    st.warning("조건에 맞는 여행지가 없습니다. 필터를 조정해 주세요.")
else:
    # (v26과 동일)
    top5_df = filtered_df.head(5)
    for _, row in top5_df.iterrows():
        with st.container(border=True):
            col_score, col_info, col_cost, col_weather = st.columns([1, 3, 2, 2])
            with col_score:
                st.metric(label="추천 점수", value=f"{row['Score']}/100")
            with col_info:
                st.subheader(f"📍 {row['City']}, {row['Country']}")
                st.caption(f"{row['Description']}")
                if 'Wiki_URL' in row and row['Wiki_URL']:
                    st.markdown(f"[Wikipedia에서 더 알아보기 ↗]({row['Wiki_URL']})") 
            with col_cost:
                st.metric(label="1일 예산", value=f"{row['Avg_Cost_KRW']:,} 원")
                if jpy_rate > 0:
                    jpy_cost = row['Avg_Cost_KRW'] * jpy_rate
                    st.caption(f" (약 {jpy_cost:,.0f} 엔)")
            with col_weather:
                st.metric(label="오늘의 날씨", value=f"{row['Today_Weather_Emoji']}")
        st.write("") 

# 12-2. 기술 분석 탭 (4개 탭) (v26과 동일)
st.divider()
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 ML 기술 분석 (K-Means)",
    "📋 필터링된 상세 데이터",
    "🌦️ 실시간 날씨 (API)",
    "ℹ️ About (프로젝트 정보)"
])

with tab1:
    # (v26과 동일)
    st.header("🤖 ML 기술 분석 (K-Means)")
    st.subheader("✅ 1. ML 활용: 유사 여행지 추천 (콘텐츠 기반)")
    st.markdown("K-Means로 분류된 '경제/지리적' 그룹을 기반으로 유사한 여행지를 추천합니다.")
    if not df_travel_base.empty:
        city_list_for_ml = ["-- 전체 보기 --"] + sorted(df_travel_base['City'].tolist())
        selected_city_for_ml = st.selectbox("도시를 선택하면, 해당 도시와 '유사한 그룹'만 필터링됩니다:", city_list_for_ml)
        df_ml_filtered = df_travel_base.copy()
        if selected_city_for_ml != "-- 전체 보기 --":
            target_cluster = df_travel_base[df_travel_base['City'] == selected_city_for_ml].iloc[0]['Cluster']
            df_ml_filtered = df_travel_base[df_travel_base['Cluster'] == target_cluster]
            st.success(f"'{selected_city_for_ml}'(은)는 **{target_cluster}번 그룹**입니다. 이 그룹의 도시들만 표시합니다.")
    else:
        st.info("데이터 로딩 중...")
    st.divider()
    st.subheader("✅ 2. ML 시각화: 클러스터 맵 & 예산 분포")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("**ML 클러스터(경제/지리) 지도**")
        if df_ml_filtered.empty: st.warning("Gist 데이터를 로드하지 못했습니다.")
        else: st.map(df_ml_filtered, latitude='Latitude', longitude='Longitude', color='ML_Map_Color')
    with col2:
        st.markdown("**예산별 여행지 분포 (Bar Chart)**")
        if not df_ml_filtered.empty and len(df_ml_filtered) > 0:
            min_data = int(df_ml_filtered['Avg_Cost_KRW'].min()); min_val = (min_data // 100000) * 100000 
            max_val = int(df_ml_filtered['Avg_Cost_KRW'].max()); max_bin = (max_val // 100000 + 1) * 100000 
            if max_bin == 0: max_bin = 100000 
            bin_edges = list(range(min_val, max_bin + 100000, 100000))
            if not bin_edges: bin_edges = [min_val, max_bin]
            bin_labels = [];
            for i in range(len(bin_edges) - 1):
                start_label = f"{bin_edges[i]//10000}만"; end_label = f"{bin_edges[i+1]//10000}만"
                if i == len(bin_edges) - 2: bin_labels.append(f"{start_label} 이상")
                else: bin_labels.append(f"{start_label}~{end_label}")
            if not bin_labels: bin_labels = [f"{min_val}~{max_bin}"] 
            bins = pd.cut(df_ml_filtered['Avg_Cost_KRW'], bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)
            df_ml_filtered['Budget_Bin'] = bins.astype(str)
            plot_df = df_ml_filtered.groupby(['Budget_Bin', 'Cluster']).agg(Count=('City', 'size'), Cities=('City', lambda x: ', '.join(x)), Countries=('Country', 'unique')).reset_index()
            fig = px.bar(plot_df, x='Budget_Bin', y='Count', color='Cluster', color_discrete_map={str(i): color for i, color in enumerate(ML_CLUSTER_COLORS)}, hover_data=['Cities', 'Countries'], title="선택된 여행지의 예산 분포")
            fig.update_layout(xaxis_title="1. 평균 예산 (KRW)", yaxis_title="도시 수")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("표시할 차트 데이터가 없습니다.")
    st.divider()
    with st.expander("ℹ️ (기술 증명) ML 모델 3D 시각화 및 최적화 (Elbow Method)"):
        if not df_travel_base.empty:
            st.markdown("#### 3D 클러스터링 시각화 (3D Scatter Plot)")
            fig_3d = px.scatter_3d(df_travel_base, x='Longitude', y='Latitude', z='Avg_Cost_KRW', color='Cluster', hover_name='City', color_discrete_map={str(i): color for i, color in enumerate(ML_CLUSTER_COLORS)}, title="K-Means 3D 클러스터링 결과")
            fig_3d.update_layout(scene = dict(zaxis = dict(title='1일 예산 (KRW)')))
            st.plotly_chart(fig_3d, use_container_width=True)
            st.divider()
            st.markdown("#### K-Means 최적화 (Elbow Method)")
            elbow_features = df_travel_base[['Avg_Cost_KRW', 'Latitude', 'Longitude']]
            elbow_df = calculate_elbow_data(elbow_features)
            if elbow_df is not None:
                fig_elbow = px.line(elbow_df, x='K (클러스터 수)', y='Inertia (응집도)', title="K값에 따른 Inertia 변화 (엘보우 메소드)", markers=True)
                fig_elbow.update_traces(marker=dict(size=8))
                st.plotly_chart(fig_elbow, use_container_width=True)
            else: st.warning("데이터가 너무 적어 엘보우 차트를 생성할 수 없습니다.")
        else: st.info("먼저 Gist 데이터를 로드해주세요.")

with tab2:
    # (v26과 동일)
    st.subheader("📋 필터링된 여행지 상세 데이터 (Score 기준 정렬)")
    st.markdown("사이드바의 필터 조건에 따라 실시간으로 필터링되며, 추천 점수(Score)가 높은 순으로 정렬됩니다.")
    if not filtered_df.empty:
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    else:
        st.warning("조건에 맞는 여행지가 없습니다.")

with tab3:
    # (v26과 동일)
    st.subheader("🌦️ 7일간 실시간 날씨 (Live API)")
    st.markdown("출발 1주일 전, 이 탭에서 도시의 실시간 날씨를 확인할 수 있습니다.")
    if df_travel_base.empty:
        st.warning("Gist 데이터를 로드하지 못했습니다.")
    else:
        st.markdown("**일본 전역 '오늘' 날씨 지도**")
        st.map(df_travel_base, latitude='Latitude', longitude='Longitude', color='Today_Weather_Color')
        st.caption(f"☀️(맑음) 🌧️(비) ❄️(눈) ☁️(흐림)")
        st.divider()
        all_city_list = sorted(df_travel_base['City'].tolist())
        selected_city_for_weather = st.selectbox("7일간 상세 예보를 볼 도시를 선택하세요:", all_city_list, key="weather_city_select")
        if selected_city_for_weather:
            city_info = df_travel_base[df_travel_base['City'] == selected_city_for_weather].iloc[0]
            lat = city_info['Latitude']; lon = city_info['Longitude']
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
                        st.metric(label="최고/최저", value=f"{forecast_data['temperature_2m_max'][i]}°C", delta=f"{forecast_data['temperature_2m_min'][i]}°C", delta_color="off")
            else: st.error("날씨 정보를 불러오는 데 실패했습니다.")

with tab4:
    # (v26과 동일)
    st.header("ℹ️ About 이 프로젝트")
    st.markdown("이 대시보드는 머신러닝 공부 및 구현을 위해 제작된 **'포트폴리오'**입니다.")
    st.markdown("---")
    st.subheader("1. 프로젝트 목적")
    st.markdown("1. **데이터 융합:** Gist DB(정적)와 Meteo API(동적)의 데이터를 융합")
    st.markdown("2. **머신러닝 적용:** 머신러닝 지식을 활용한 **군집화(K-Means)** 및 **기술적 검증(Elbow Method, 3D Plot)** 구현")
    st.markdown("3. **데이터 시각화:** API와 ML 모델의 결과를 **직관적인 UI(카드, 탭, 2종류 맵)**로 시각화")
    st.markdown("4. **빠른 프로토타이핑:** Streamlit을 활용한 신속한 아이디어 구현 및 배포")
    st.markdown("---")
    st.subheader("2. 사용한 핵심 기술 스택")
    st.code("Python, Streamlit (Front-end), Pandas, Scikit-learn (K-Means, StandardScaler), Plotly (2D/3D)", language="python")
    st.markdown("---")
    st.subheader("3. 데이터 출처 (100% API 기반)")
    st.markdown("- **여행지 DB (Gist API):** 본인이 직접 구축한 JSON 데이터를 GitHub Gist에 배포하여 API로 활용")
    st.markdown("- **실시간 날씨 (Meteo API):** Open-Meteo의 7일 예보(Forecast) API 실시간 호출")
