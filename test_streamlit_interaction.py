from streamlit.testing.v1 import AppTest


INDIVIDUAL = "🎯 Análisis Individual por Partido"


def main():
    at = AppTest.from_file("app_mlb.py", default_timeout=120).run()
    assert len(at.exception) == 0, [str(x.value) for x in at.exception]

    # Force individual mode and rerun, reproducing a normal sidebar interaction.
    if len(at.sidebar.radio):
        at.sidebar.radio[0].set_value(INDIVIDUAL).run(timeout=120)
        assert len(at.exception) == 0, [str(x.value) for x in at.exception]

    # The first main selectbox is the game selector when a slate exists.
    if len(at.main.selectbox):
        game_box = at.main.selectbox[0]
        if len(game_box.options) > 1:
            game_box.select_index(1).run(timeout=120)
            assert len(at.exception) == 0, [str(x.value) for x in at.exception]

    # Click the individual-analysis button when present. Missing odds may lead to
    # a user-visible st.error/st.stop, which is acceptable; uncaught exceptions are not.
    target = None
    for button in at.main.button:
        if "Simulación" in button.label or "Simulacion" in button.label:
            target = button
            break
    if target is not None:
        target.click().run(timeout=120)
        assert len(at.exception) == 0, [str(x.value) for x in at.exception]

    print("Streamlit interaction smoke passed")


if __name__ == "__main__":
    main()
