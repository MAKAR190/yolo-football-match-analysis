from modules.stats.streamlit_app import StatsStreamlitApp


def main():
    app = StatsStreamlitApp(stats_root="outputs/stats")
    app.run()


if __name__ == "__main__":
    main()

