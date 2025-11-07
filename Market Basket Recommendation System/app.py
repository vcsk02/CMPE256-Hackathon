# app.py
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import streamlit as st
import pandas as pd

# ---------- helpers ----------
def _clean_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip()
    if " (SKU:" in x:
        x = x.split(" (SKU:")[0]
    x = " ".join(x.split()).lower()
    return x

def load_model(pkl_path: str = "recommendation_engine.pkl"):
    p = Path(pkl_path)
    if not p.exists():
        st.error(f"Model file not found: {p.resolve()}")
        st.stop()
    with open(p, "rb") as f:
        data = pickle.load(f)
    return (
        data.get("rules", None),
        data.get("frequent_itemsets", None),
        data.get("products", []),
        data.get("cooccurrence_item_counts", None),
        data.get("cooccurrence_pair_counts", None),
    )

class SimpleRecommendationEngine:
    def __init__(self, rules_df, item_counts=None, pair_counts=None):
        self.rules = rules_df
        self.item_counts = item_counts
        self.pair_counts = pair_counts

    def get_recommendations(
        self, cart_items: List[str], top_n: int = 5, min_confidence: float = 0.3, min_lift: float = 1.0
    ) -> List[Dict]:
        if not cart_items:
            return []

        cart_set_lower = {_clean_name(c) for c in cart_items if isinstance(c, str) and c.strip()}

        # ------- rule-based -------
        rule_recs = []
        if self.rules is not None and hasattr(self.rules, "empty") and not self.rules.empty:
            agg = defaultdict(lambda: {"confidence": 0.0, "lift": 0.0, "support": 0.0, "count": 0})
            for _, rule in self.rules.iterrows():
                ants_lower = {_clean_name(s) for s in rule["antecedents"]}
                cons_lower = {_clean_name(s) for s in rule["consequents"]}
                if ants_lower.issubset(cart_set_lower):
                    new_items_lower = cons_lower - cart_set_lower
                    if new_items_lower and rule["confidence"] >= min_confidence and rule["lift"] >= min_lift:
                        for item_lower in new_items_lower:
                            item_original = next(
                                (orig for orig in rule["consequents"] if _clean_name(orig) == item_lower), item_lower
                            )
                            agg[item_original]["confidence"] = max(agg[item_original]["confidence"], float(rule["confidence"]))
                            agg[item_original]["lift"] = max(agg[item_original]["lift"], float(rule["lift"]))
                            agg[item_original]["support"] += float(rule["support"])
                            agg[item_original]["count"] += 1

            if agg:
                for item, s in agg.items():
                    score = s["confidence"] * 0.5 + s["lift"] * 0.3 + s["support"] * 0.2
                    rule_recs.append({
                        "product": item,
                        "confidence": round(s["confidence"], 3),
                        "lift": round(s["lift"], 3),
                        "support": round(s["support"], 4),
                        "score": round(score, 3),
                        "frequency": s["count"],
                        "source": "rules"
                    })
                rule_recs.sort(key=lambda x: x["score"], reverse=True)
                return rule_recs[:top_n]

        # ------- co-occurrence fallback -------
        if self.item_counts and self.pair_counts:
            scores = defaultdict(lambda: {"pair": 0, "den": 0})

            # map clean -> original
            name_map = {}
            for it in self.item_counts.keys():
                name_map[_clean_name(it)] = it

            cart_known = [c for c in cart_set_lower if c in name_map]
            if not cart_known:
                return []

            for c in cart_known:
                c_orig = name_map[c]
                c_count = self.item_counts.get(c_orig, 0)
                if not c_count:
                    continue
                for (a, b), pcount in self.pair_counts.items():
                    if a == c_orig or b == c_orig:
                        other = b if a == c_orig else a
                        if _clean_name(other) in cart_set_lower:
                            continue
                        scores[other]["pair"] += pcount
                        scores[other]["den"] += c_count

            if not scores:
                return []

            recs = []
            total_items = max(1, sum(self.item_counts.values()))
            for item, s in scores.items():
                conf_like = s["pair"] / s["den"] if s["den"] else 0.0
                sup_like = self.item_counts.get(item, 0) / total_items
                combined = 0.7 * conf_like + 0.3 * sup_like
                recs.append({
                    "product": item,
                    "confidence": round(conf_like, 3),
                    "lift": None,
                    "support": round(sup_like, 4),
                    "score": round(combined, 3),
                    "frequency": self.item_counts.get(item, 0),
                    "source": "cooccurrence"
                })
            recs.sort(key=lambda x: x["score"], reverse=True)
            return recs[:top_n]

        return []

# ---------- UI ----------
st.set_page_config(page_title="Market Basket Recommender", page_icon="🛒", layout="wide")
st.title("🛒 Market Basket Recommendation System")
st.caption("Local UI • uses your mined rules and co-occurrence fallback (no static data)")

with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Top N", 1, 20, 5, 1)
    min_conf = st.slider("Min Confidence (rules)", 0.0, 1.0, 0.3, 0.01)
    min_lift = st.slider("Min Lift (rules)", 0.0, 3.0, 1.0, 0.05)
    model_path = st.text_input("Model file (pkl)", "recommendation_engine.pkl")
    st.markdown("---")
    st.caption("Tip: enter multiple items separated by commas.")
    if st.button("Use example"):
        st.session_state["cart_text"] = "Axis P3225-LVE Network Camera (SKU: 0935-001)"

st.subheader("🧺 Enter cart items")
cart_text = st.text_input("Items (comma-separated):", value=st.session_state.get("cart_text", ""))

# load model + engine
rules, frequent_itemsets, products, item_counts, pair_counts = load_model(model_path)
engine = SimpleRecommendationEngine(rules, item_counts=item_counts, pair_counts=pair_counts)

col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("#### Products (from model)")
    st.write(f"Known products: {len(products)}")
    if products:
        st.dataframe(pd.DataFrame(sorted(products), columns=["product"]).head(50), use_container_width=True)

with col2:
    st.markdown("#### Mining summary")
    st.write(f"Frequent itemsets: {0 if frequent_itemsets is None else len(frequent_itemsets)}")
    st.write(f"Rules: {0 if rules is None else len(rules)}")

st.markdown("---")

if st.button("🔎 Get Recommendations"):
    cart = [c.strip() for c in cart_text.split(",") if c.strip()]
    if not cart:
        st.warning("Add at least one cart item.")
    else:
        recs = engine.get_recommendations(cart, top_n=top_n, min_confidence=min_conf, min_lift=min_lift)
        if not recs:
            st.info("No recommendations for this combination with current thresholds.")
        else:
            df = pd.DataFrame(recs)
            st.success(f"Found {len(df)} recommendations")
            st.dataframe(df, use_container_width=True)
