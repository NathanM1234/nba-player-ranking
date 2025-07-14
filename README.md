
````markdown
# NBA Player Ranking App

This is a Streamlit web application that ranks NBA players based on various metrics including box score statistics, advanced stats, defensive/offensive impact, era adjustments, accolades, and more. Users can view rankings for both career and individual seasons, filter players by various criteria, and download the data as CSV.

---

## Features

- **Career and Season mode** rankings
- Composite scores built from multiple stats and advanced metrics
- Filters by player name, team, position, height, and other attributes
- Ability to toggle which metrics to include in the rankings
- Export rankings as downloadable CSV files
- Interactive Streamlit web interface

---

## Getting Started

### Prerequisites

- Python 3.10
- Git

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/nba-player-ranking.git
   cd nba-player-ranking
````

2. (Optional) Create and activate a virtual environment:

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Run the App Locally

```bash
streamlit run nba_ranker_app.py
```

---

## Usage

* Select **Career** or **Season** mode
* Use the filters on the sidebar to search and narrow down players
* Toggle which metrics to include in the composite score
* View the ranked player list and download CSV files of the rankings

---

## Deployment

You can deploy this app easily on [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting your GitHub repo and selecting `nba_ranker_app.py` as the main script.

---

## Dependencies

* `streamlit`
* `nba_api`
* `pandas`
* `numpy`
* `beautifulsoup4`
* `requests`

---

## Contributing

Contributions are welcome! Please open issues or pull requests.

---

## Author

Nathan Monroe

```

```
