import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for pkg in ["nba_api", "pandas", "numpy", "streamlit", "beautifulsoup4", "requests", "altair"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import pandas as pd
import numpy as np
import streamlit as st
import time
import random
import altair as alt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, playergamelog
import requests
from bs4 import BeautifulSoup

st.set_page_config(layout="wide")

def height_to_inches(h):
    try:
        f, i = h.split('-')
        return int(f)*12 + int(i)
    except:
        return None

@st.cache_data(show_spinner=False)
def cached_fetch_player_accolades(player_name):
    try:
        search_url = f"https://www.basketball-reference.com/search/search.fcgi?search={player_name.replace(' ', '+')}"
        res = requests.get(search_url)
        soup = BeautifulSoup(res.text, 'html.parser')

        player_link = None
        for div in soup.find_all('div', {'class': 'search-item-url'}):
            href = div.text.strip()
            if href.startswith("/players/"):
                player_link = "https://www.basketball-reference.com" + href
                break
        if not player_link:
            return {'MVPs':0, 'AllNBA':0, 'AllDef':0, 'Championships':0}

        time.sleep(1)
        res = requests.get(player_link)
        soup = BeautifulSoup(res.text, 'html.parser')

        # Count Championships
        champ_count = 0
        highlights_div = soup.find('div', id='all_highlights')
        if highlights_div:
            # Extract text from comments or inside div (some data may be in comments)
            highlights_text = highlights_div.get_text().lower()
            champ_count = highlights_text.count("nba champion")

        # Parse awards table if present
        awards_table = soup.find('table', id='awards')
        mvp_count = 0
        allnba_count = 0
        alldef_count = 0

        if awards_table:
            rows = awards_table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if not cols:
                    continue
                award = cols[0].get_text().lower()
                count_text = cols[1].get_text() if len(cols) > 1 else "1"
                count = int(count_text) if count_text.isdigit() else 1

                if 'most valuable player' in award:
                    mvp_count += count
                elif 'all-nba' in award:
                    allnba_count += count
                elif 'all-defensive' in award:
                    alldef_count += count

        return {
            'MVPs': mvp_count,
            'AllNBA': allnba_count,
            'AllDef': alldef_count,
            'Championships': champ_count
        }
    except Exception as e:
        print(f"Error scraping accolades for {player_name}: {e}")
        return {'MVPs':0, 'AllNBA':0, 'AllDef':0, 'Championships':0}

def calculate_era_factor(year_start):
    if year_start < 1980: 
        return 0.95
    elif year_start < 1995: 
        return 1.0
    else: 
        return 1.05

def fetch_defensive_pressure(player_id, season=None):
    # Placeholder for actual data, simulated for now
    return random.uniform(0.8, 1.2)

def fetch_opponent_quality(player_id, season=None):
    # Placeholder for actual data, simulated for now
    return random.uniform(0.8, 1.2)

@st.cache_data(show_spinner=False)
def cached_fetch_all_players():
    # Merge active and inactive players; filter since 1976-77 (merger season)
    all_players = players.get_active_players() + players.get_inactive_players()
    # Add team, height, position if missing - may require external source or set None
    for p in all_players:
        p.setdefault('team', None)
        p.setdefault('height', None)
        p.setdefault('position', None)
    return all_players

@st.cache_data(show_spinner=False)
def cached_fetch_career_stats(player):
    pid = player['id']
    try:
        career_df = playercareerstats.PlayerCareerStats(player_id=pid).get_data_frames()[0]
        if career_df.empty:
            return None
        career_totals = career_df.iloc[-1]
        gp = career_totals.get("GP", 1)
        season_id = career_df['SEASON_ID'].iloc[0] if 'SEASON_ID' in career_df.columns else "2000"
        year_start = int(season_id[:4])
        stats = {
            "PlayerID": pid,
            "PlayerName": player['full_name'],
            "PTS_avg": career_totals.get("PTS", 0) / gp,
            "REB_avg": career_totals.get("REB", 0) / gp,
            "AST_avg": career_totals.get("AST", 0) / gp,
            "STL_avg": career_totals.get("STL", 0) / gp,
            "BLK_avg": career_totals.get("BLK", 0) / gp,
            "PTS_tot": career_totals.get("PTS", 0),
            "REB_tot": career_totals.get("REB", 0),
            "AST_tot": career_totals.get("AST", 0),
            "STL_tot": career_totals.get("STL", 0),
            "BLK_tot": career_totals.get("BLK", 0),
            "GP": gp,
            "YearStart": year_start,
            "HEIGHT": player.get("height", None),
            "POSITION": player.get("position", None),
            "TEAM": player.get("team", None),
        }
        accolades = cached_fetch_player_accolades(player['full_name'])
        stats.update(accolades)
        stats['EraFactor'] = calculate_era_factor(stats['YearStart'])
        stats['DefensivePressure'] = fetch_defensive_pressure(pid)
        stats['OpponentQuality'] = fetch_opponent_quality(pid)
        return stats
    except Exception as e:
        print(f"Error fetching career stats for {player['full_name']}: {e}")
        return None

@st.cache_data(show_spinner=False)
def cached_fetch_season_stats(player):
    pid = player['id']
    try:
        gamelog_df = playergamelog.PlayerGameLog(player_id=pid, season_type_all_star='Regular Season').get_data_frames()[0]
        if gamelog_df.empty:
            return None
        season_stats_list = []
        for season, season_df in gamelog_df.groupby("SEASON_ID"):
            gp = len(season_df)
            year_start = int(season[:4])
            stats = {
                "PlayerID": pid,
                "PlayerName": player['full_name'],
                "Season": season,
                "PTS_avg": season_df["PTS"].mean(),
                "REB_avg": season_df["REB"].mean(),
                "AST_avg": season_df["AST"].mean(),
                "STL_avg": season_df["STL"].mean(),
                "BLK_avg": season_df["BLK"].mean(),
                "PTS_tot": season_df["PTS"].sum(),
                "REB_tot": season_df["REB"].sum(),
                "AST_tot": season_df["AST"].sum(),
                "STL_tot": season_df["STL"].sum(),
                "BLK_tot": season_df["BLK"].sum(),
                "GP": gp,
                "YearStart": year_start,
                "HEIGHT": player.get("height", None),
                "POSITION": player.get("position", None),
                "TEAM": player.get("team", None),
            }
            accolades = cached_fetch_player_accolades(player['full_name'])
            stats.update(accolades)
            stats['EraFactor'] = calculate_era_factor(stats['YearStart'])
            stats['DefensivePressure'] = fetch_defensive_pressure(pid, season)
            stats['OpponentQuality'] = fetch_opponent_quality(pid, season)
            season_stats_list.append(stats)
        return pd.DataFrame(season_stats_list)
    except Exception as e:
        print(f"Error fetching season stats for {player['full_name']}: {e}")
        return None

class NBAPlayerRanker:
    def __init__(self, mode="career"):
        self.mode = mode
        self.df = None

    def build_dataset(self):
        all_players = cached_fetch_all_players()
        player_data = []
        season_data = []
        st.info(f"Fetching data for {len(all_players)} players... This may take several minutes.")
        for idx, player in enumerate(all_players):
            if self.mode == "career":
                stats = cached_fetch_career_stats(player)
                if stats:
                    player_data.append(stats)
            else:
                season_df = cached_fetch_season_stats(player)
                if season_df is not None and not season_df.empty:
                    season_data.append(season_df)
            if idx % 25 == 0 and idx > 0:
                st.write(f"Processed {idx} players...")

        if self.mode == "career":
            self.df = pd.DataFrame(player_data)
        else:
            self.df = pd.concat(season_data, ignore_index=True) if season_data else pd.DataFrame()

    def normalize(self, col):
        if col not in self.df.columns or self.df[col].isnull().all():
            self.df[col + "_norm"] = 0
            return
        min_val = self.df[col].min()
        max_val = self.df[col].max()
        if max_val - min_val == 0:
            self.df[col + "_norm"] = 0
            return
        self.df[col + "_norm"] = (self.df[col] - min_val) / (max_val - min_val)

    def compute_composite_score(self, weights):
        for col in weights.keys():
            base_col = col.replace('_norm', '')
            if base_col in self.df.columns:
                self.normalize(base_col)
            else:
                self.df[col] = 0
                self.df[col + "_norm"] = 0

        self.df['composite_score'] = 0
        for col, w in weights.items():
            if col in self.df.columns:
                self.df['composite_score'] += self.df[col].fillna(0) * w
            elif col + '_norm' in self.df.columns:
                self.df['composite_score'] += self.df[col + '_norm'].fillna(0) * w
        return self.df

    def export_to_csv(self, filename):
        if 'composite_score' not in self.df.columns:
            st.warning("Please compute composite score before export.")
            return
        self.df.to_csv(filename, index=False)
        st.success(f"Exported rankings to {filename}")

st.title("🏀 NBA Player Ranking with Accolades, Season Mode, Comparison, and Visualizations")

mode = st.radio("Select Mode:", ["career", "season"], index=0)

if 'weights' not in st.session_state:
    st.session_state.weights = {
        'PTS_avg_norm': 0.2,
        'REB_avg_norm': 0.15,
        'AST_avg_norm': 0.15,
        'STL_avg_norm': 0.1,
        'BLK_avg_norm': 0.1,
        'PTS_tot_norm': 0.05,
        'REB_tot_norm': 0.05,
        'AST_tot_norm': 0.05,
        'STL_tot_norm': 0.025,
        'BLK_tot_norm': 0.025,
        'MVPs_norm': 0.025,
        'AllNBA_norm': 0.025,
        'AllDef_norm': 0.025,
        'Championships_norm': 0.025,
        'EraFactor_norm': 0.05,
        'DefensivePressure_norm': 0.025,
        'OpponentQuality_norm': 0.025
    }

with st.sidebar:
    st.header("Filters")
    team_filter = st.text_input("Filter by Team (partial):")
    position_filter = st.text_input("Filter by Position (partial):")
    name_filter = st.text_input("Search Player Name:")
    min_height = st.number_input("Min Height (inches):", 60, 90, 60)
    max_height = st.number_input("Max Height (inches):", 60, 90, 90)

    st.header("Adjust Weights")
    for key in st.session_state.weights.keys():
        st.session_state.weights[key] = st.slider(
            key.replace("_norm", "").replace("_", " ").title(),
            0.0, 1.0,
            st.session_state.weights[key],
            0.01
        )

    if st.button("Reset Filters and Weights"):
        team_filter = position_filter = name_filter = ""
        min_height, max_height = 60, 90
        for key in st.session_state.weights.keys():
            st.session_state.weights[key] = 0.05

ranker = NBAPlayerRanker(mode=mode)

if st.button("Generate Rankings"):
    with st.spinner("Fetching data and computing rankings..."):
        ranker.build_dataset()

        if ranker.df is None or ranker.df.empty:
            st.warning("No data fetched. Please try again or check your filters.")
            st.stop()

        # Height conversion
        if 'HEIGHT' in ranker.df.columns:
            ranker.df['Height_in'] = ranker.df['HEIGHT'].apply(height_to_inches)
        else:
            ranker.df['Height_in'] = None

        df_filtered = ranker.df.copy()

        # Apply filters
        if team_filter:
            df_filtered = df_filtered[df_filtered['TEAM'].str.contains(team_filter, case=False, na=False)]
        if position_filter:
            df_filtered = df_filtered[df_filtered['POSITION'].str.contains(position_filter, case=False, na=False)]
        if name_filter:
            df_filtered = df_filtered[df_filtered['PlayerName'].str.contains(name_filter, case=False, na=False)]
        if df_filtered['Height_in'].notnull().any():
            df_filtered = df_filtered[(df_filtered['Height_in'] >= min_height) & (df_filtered['Height_in'] <= max_height)]

        ranker.df = df_filtered

        # Season selector if season mode
        if mode == "season" and not ranker.df.empty:
            seasons = sorted(ranker.df['Season'].unique())
            selected_season = st.sidebar.selectbox("Select Season", options=["All"] + seasons)
            if selected_season != "All":
                ranker.df = ranker.df[ranker.df['Season'] == selected_season]

        # Normalize accolade and factor columns
        for col in ['MVPs', 'AllNBA', 'AllDef', 'Championships']:
            if col in ranker.df.columns:
                ranker.normalize(col)
                ranker.df[col + '_norm'] = ranker.df[col + '_norm'].fillna(0)
            else:
                ranker.df[col + '_norm'] = 0
        for col in ['EraFactor', 'DefensivePressure', 'OpponentQuality']:
            if col in ranker.df.columns:
                ranker.normalize(col)
                ranker.df[col + '_norm'] = ranker.df[col + '_norm'].fillna(1)
            else:
                ranker.df[col + '_norm'] = 1

        ranker.compute_composite_score(st.session_state.weights)

        st.success(f"Computed rankings for {len(ranker.df)} players.")
        st.dataframe(ranker.df.sort_values('composite_score', ascending=False), use_container_width=True)

        # Player comparison UI
        player_names = ranker.df['PlayerName'].unique().tolist()
        selected_players = st.multiselect("Select Players to Compare", options=player_names)
        if selected_players:
            comp_df = ranker.df[ranker.df['PlayerName'].isin(selected_players)]
            st.write("### Player Comparison")
            st.dataframe(comp_df.sort_values('composite_score', ascending=False), use_container_width=True)

            # If in season mode, show line chart for composite score across seasons
            if mode == "season":
                line_chart_data = comp_df.pivot(index="Season", columns="PlayerName", values="composite_score").reset_index()
                line_chart_data = line_chart_data.melt('Season', var_name='PlayerName', value_name='Composite Score')
                chart = alt.Chart(line_chart_data).mark_line(point=True).encode(
                    x="Season",
                    y="Composite Score",
                    color='PlayerName:N'
                ).properties(width=800, height=400)
                st.altair_chart(chart, use_container_width=True)

        # Visualizations
        st.write("### Composite Score Distribution")
        hist = alt.Chart(ranker.df).mark_bar().encode(
            alt.X("composite_score", bin=alt.Bin(maxbins=50)),
            y='count()'
        ).properties(width=800)
        st.altair_chart(hist, use_container_width=True)

        st.write("### Composite Score vs MVPs")
        scatter = alt.Chart(ranker.df).mark_circle(size=60).encode(
            x="MVPs",
            y="composite_score",
            tooltip=["PlayerName", "composite_score", "MVPs"]
        ).interactive()
        st.altair_chart(scatter, use_container_width=True)

        # Export CSV
        filename = f"nba_player_rankings_{mode}.csv"
        ranker.export_to_csv(filename)
        with open(filename, "rb") as f:
            st.download_button("Download Rankings CSV", f, file_name=filename, mime="text/csv")
