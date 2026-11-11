# Dark-store random-demand simulator

Run: double-click `run_dashboard.bat`, or from this folder `py -m streamlit run app.py`, then open http://localhost:8501.
If you edit `engine.py`, stop the dashboard (Ctrl+C) and start it again; Streamlit does not reload it.

- `engine.py`: Monte-Carlo demand, store selection (exact MILP or greedy), cost model, what-if analyses
- `app.py`: Streamlit dashboard
- Reads `../data` (hexagons, candidate sites, travel times, competitor stores) and `../tableau` (hex polygons)
