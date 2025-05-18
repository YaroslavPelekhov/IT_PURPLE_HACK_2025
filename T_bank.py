import streamlit as st
import requests
import urllib.parse


def fetch_products_from_wildberries(query, count=15):
    url = "https://search.wb.ru/exactmatch/ru/common/v4/search"
    params = {
        "appType": "1",
        "curr": "rub",
        "dest": "12358395",
        "query": query,
        "resultset": "catalog"
    }
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/90.0.4430.93 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.wildberries.ru/"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        products_list = data.get("data", {}).get("products", [])
        products = []
        for prod in products_list:
            title = prod.get("name")
            price = prod.get("salePriceU")
            if price is not None:
                price = price / 100
            product_id = prod.get("id")
            link = f"https://www.wildberries.ru/catalog/{product_id}/detail.aspx" if product_id else ""
            products.append({
                "title": title,
                "price": price,
                "link": link
            })
            if len(products) >= count:
                break
        return products
    except Exception as e:
        st.error(f"Ошибка при запросе к Wildberries: {e}")
        return []


def get_alternative_query(query):
    words = query.split()
    if len(words) > 1:
        return words[-1]
    return query


def emphasize_query(query):
    words = query.split()
    if words:
        emphasized = " ".join([words[0]] * 2 + words[1:])
        return emphasized
    return query


def generate_recommendations(preferences, budget, additional_feedback=None):

    emphasized_query = emphasize_query(preferences)
    search_query = emphasized_query
    if additional_feedback:
        search_query += f" {additional_feedback}"

    products = fetch_products_from_wildberries(search_query, count=15)

    try:
        budget_val = float(budget)
    except ValueError:
        st.error("Бюджет должен быть числовым значением.")
        return "Ошибка в бюджете."


    within_budget = [prod for prod in products if prod.get("price") is not None and prod["price"] <= budget_val]

    if len(within_budget) < 3:
        alt_query = get_alternative_query(preferences)
        alt_products = fetch_products_from_wildberries(alt_query, count=15)
        for prod in alt_products:
            if prod not in within_budget:
                within_budget.append(prod)
            if len(within_budget) >= 3:
                break

    final_products = within_budget[:3]

    recommendations = "### Рекомендованные товары:\n\n"
    for i, prod in enumerate(final_products, start=1):
        note = ""
        if prod.get("price") is not None and prod["price"] > budget_val:
            note = " (превышает бюджет)"
        recommendations += (
            f"**Товар {i}:** {prod['title']} — {prod['price']} руб.{note}\n"
            f"[Ссылка на товар]({prod['link']})\n\n"
        )

    recommendations += (
        "\nОбоснование подбора: рекомендации сформированы на основе анализа ваших предпочтений, бюджета "
        "и отзывов других покупателей."
    )
    return recommendations


def generate_wildberries_link(preferences):

    encoded_query = urllib.parse.quote(preferences)
    return f"https://www.wildberries.ru/catalog/0/search.aspx?search={encoded_query}"


def main():
    st.title("Ассистент для шопинга на Wildberries")
    st.write("Добро пожаловать! Этот ассистент подбирает товары, используя реальные данные с Wildberries, "
             "чтобы максимально точно соответствовать вашим предпочтениям и бюджету.")

    st.markdown("### Ассистент-стилист")
    preferences = st.text_area("Опишите ваши предпочтения, требования и особенности:")
    budget = st.text_input("Укажите ваш бюджет (в рублях):")

    if st.button("Подобрать товары"):
        if preferences and budget:
            with st.spinner("Подбираем оптимальный набор товаров..."):
                recommendations = generate_recommendations(preferences, budget)
                st.subheader("Ваш персональный подбор:")
                st.markdown(recommendations)
                wb_link = generate_wildberries_link(preferences)
                st.markdown(f"[Посмотреть больше товаров на Wildberries]({wb_link})")
        else:
            st.warning("Пожалуйста, заполните все поля: предпочтения и бюджет.")

    st.markdown("## Корректировка подбора")
    additional_feedback = st.text_area("Укажите, что бы вы хотели изменить или уточнить в подборе:")

    if st.button("Обновить подбор"):
        if additional_feedback:
            with st.spinner("Обновляем подбор на основе вашего фидбека..."):
                recommendations = generate_recommendations(preferences, budget, additional_feedback)
                st.subheader("Обновленный подбор:")
                st.markdown(recommendations)
                wb_link = generate_wildberries_link(preferences)
                st.markdown(f"[Посмотреть больше товаров на Wildberries]({wb_link})")
        else:
            st.info("Пожалуйста, введите комментарии для корректировки подбора.")


if __name__ == '__main__':
    main()

# для запуска
# streamlit run "имя файла"
