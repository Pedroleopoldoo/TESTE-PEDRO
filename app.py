import streamlit as st
from predict import recommend

st.title("🛒 Recomendador de Produtos")

st.write("Digite os produtos que você já comprou:")

input_text = st.text_area("Produtos (separados por vírgula)")

if st.button("Recomendar"):

    history = [x.strip() for x in input_text.split(",") if x.strip()]

    if len(history) == 0:
        st.warning("Digite pelo menos um produto.")
    else:
        results = recommend(history)

        st.subheader("Recomendações:")

        for item, score in results:
            st.write(f"{item} — {score:.4f}")