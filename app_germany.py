# app_germany.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Car Advisor Germany", page_icon="🚗", layout="wide")

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
DATA_FILE = Path("data/cars_db_germany.csv")
if not DATA_FILE.exists():
    st.error("Dataset not found. Please run cars_db_germany_builder.py to create data/cars_db_germany.csv")
    st.stop()

df = pd.read_csv(DATA_FILE)

# Clean column names
df.columns = [c.strip() for c in df.columns]


# ------------------------------------------------------------
# Helper Functions for the German Market
# ------------------------------------------------------------

def compute_budget_eur(yearly_salary_eur):
    """Safe = 20% of salary, Stretch = 35%"""
    safe = yearly_salary_eur * 0.20
    stretch = yearly_salary_eur * 0.35
    return safe, stretch


def monthly_payment(principal, annual_rate_pct, months):
    if months <= 0:
        return 0
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** (-months))


def amortization_schedule(principal, rate_pct, months):
    r = rate_pct / 100 / 12
    monthly = monthly_payment(principal, rate_pct, months)
    schedule = []
    balance = principal

    for m in range(1, months + 1):
        interest = balance * r
        principal_paid = monthly - interest
        balance = max(0, balance - principal_paid)

        schedule.append({
            "month": m,
            "payment": monthly,
            "interest": interest,
            "principal": principal_paid,
            "balance": balance
        })
    return schedule


def maintenance_estimate_germany(make, year, mileage_km, fuel_type):
    """Germany-specific maintenance cost estimate (rough heuristic)."""
    base = 600
    make_low = make.lower()

    high_cost = ["bmw", "mercedes", "audi"]
    low_cost = ["toyota", "honda", "mazda", "skoda"]

    if any(m in make_low for m in high_cost):
        mult = 1.7
    elif any(m in make_low for m in low_cost):
        mult = 0.9
    else:
        mult = 1.1

    # Age factor
    if pd.notna(year):
        age = datetime.now().year - int(year)
        age_factor = 1 + max(0, age - 3) * 0.08
    else:
        age_factor = 1.2

    # Mileage factor
    if pd.notna(mileage_km):
        mile_factor = 1 + max(0, (mileage_km - 30000) / 200000)
    else:
        mile_factor = 1.0

    # Electric cars cheaper on maintenance
    if "elect" in str(fuel_type).lower():
        mult *= 0.8

    return int(base * mult * age_factor * mile_factor)


def insurance_estimate_germany(driver_age, make, model, price_eur, no_claims_years=5):
    """Very simplified insurance premium estimator."""
    base_haft = 300

    # Age multiplier
    if driver_age < 25:
        age_mult = 1.6
    elif driver_age > 65:
        age_mult = 1.2
    else:
        age_mult = 1.0

    # Brand multiplier (luxury brands)
    if any(b in make.lower() for b in ["bmw", "audi", "mercedes"]):
        brand_mult = 1.25
    else:
        brand_mult = 1.0

    price_factor = 1 + (price_eur / 50000)

    # No-claims discount
    disc = 1 - min(0.5, no_claims_years * 0.03)

    haft = int(base_haft * age_mult * brand_mult * price_factor * disc)
    teil = int(haft * 1.3)
    voll = int(haft * 1.9)

    return {
        "haftpflicht": haft,
        "teilkasko": teil,
        "vollkasko": voll
    }


# ------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------

st.title("🚗 Car Buying Advisor — Germany")
st.markdown("Smart filters, German insurance & maintenance costs, financing calculator, and personalized Top-3 picks.")

left, center, right = st.columns([2, 3, 2])


# ------------------------------------------------------------
# LEFT COLUMN — FILTERS
# ------------------------------------------------------------
with left:
    st.header("Filter Cars")

    price_min, price_max = st.slider(
        "Price (€)",
        int(df.price_eur.min(skipna=True)),
        int(df.price_eur.max(skipna=True)),
        (5000, 30000)
    )

    year_min, year_max = st.slider(
        "Year",
        int(df.year.min(skipna=True)),
        int(df.year.max(skipna=True)),
        (2013, 2022)
    )

    km_max = st.number_input("Max mileage (km)", value=120_000, step=1000)

    fuels = sorted(df.fuel_type.dropna().unique().tolist())
    chosen_fuels = st.multiselect("Fuel type", fuels, default=fuels)

    segments = sorted(df.segment.dropna().unique().tolist())
    chosen_segments = st.multiselect("Segment", segments, default=segments)

    sort_by = st.selectbox("Sort by", ["price_eur", "year", "mileage_km", "power_ps"])

    filtered = df[
        (df.price_eur >= price_min) &
        (df.price_eur <= price_max) &
        (df.year >= year_min) &
        (df.year <= year_max) &
        (df.mileage_km <= km_max) &
        (df.fuel_type.isin(chosen_fuels)) &
        (df.segment.isin(chosen_segments))
    ].copy()

    st.markdown(f"Found **{len(filtered):,}** matching cars")

    st.dataframe(
        filtered.sort_values(sort_by, ascending=(sort_by != "year")).head(10),
        use_container_width=True
    )

    selected_id = st.selectbox(
        "Select a car ID",
        options=filtered.id.astype(str).tolist() if len(filtered) > 0 else []
    )


# ------------------------------------------------------------
# CENTER COLUMN — CAR DETAILS
# ------------------------------------------------------------
with center:
    st.header("Car Details & Cost Analysis")

    if selected_id:
        car = filtered[filtered.id.astype(str) == selected_id].iloc[0].to_dict()

        st.subheader(f"{car['make']} {car['model']} ({int(car['year'])}) — €{int(car['price_eur']):,}")
        st.write(
            f"KM: {int(car['mileage_km']):,} km • "
            f"Fuel: {car['fuel_type']} • "
            f"Power: {car.get('power_ps', 'N/A')} PS"
        )

        # Car image
        img_url = f"https://source.unsplash.com/900x400/?{car['make']}+{car['model']}"
        st.image(img_url, use_column_width=True)

        # Maintenance
        maint = maintenance_estimate_germany(car["make"], car["year"], car["mileage_km"], car["fuel_type"])
        st.write(f"**Estimated annual maintenance (Germany): €{maint:,}**")

        # Insurance
        st.markdown("### Insurance Estimator (Germany)")
        driver_age = st.number_input("Driver age", value=35, min_value=18)
        no_claims = st.number_input("No-claims years (Schadenfreiheitsklasse)", value=5, min_value=0)
        ins = insurance_estimate_germany(driver_age, car["make"], car["model"], car["price_eur"], no_claims)

        st.write(f"**Haftpflicht:** €{ins['haftpflicht']:,}")
        st.write(f"**Teilkasko:** €{ins['teilkasko']:,}")
        st.write(f"**Vollkasko:** €{ins['vollkasko']:,}")

        st.markdown("---")

        # Financing
        st.subheader("Financing / Leasing Calculator (Germany)")
        financing_type = st.radio("Type", ["Finanzierung (Loan)", "Leasing"])
        down_pct = st.number_input("Down payment (%)", value=10.0)
        term = st.selectbox("Term (months)", [24, 36, 48, 60], index=2)
        apr = st.number_input("APR (%)", value=4.5)

        down_payment = car["price_eur"] * down_pct / 100
        loan_amount = car["price_eur"] - down_payment
        monthly = monthly_payment(loan_amount, apr, term)

        st.write(f"Down payment: €{down_payment:,.0f}")
        st.write(f"Loan amount: €{loan_amount:,.0f}")
        st.write(f"**Monthly payment: €{monthly:,.2f}**")

        if st.button("Show first 12 months schedule"):
            schedule = amortization_schedule(loan_amount, apr, term)[:12]
            st.table(pd.DataFrame(schedule))

    else:
        st.info("Please select a car from the left table.")


# ------------------------------------------------------------
# RIGHT COLUMN — TOP-3 RECOMMENDATIONS
# ------------------------------------------------------------
with right:
    st.header("Top-3 Recommendations (AI Scoring)")

    salary = st.number_input("Your yearly net salary (€)", value=48000, step=1000)
    commute_km = st.number_input("Daily commute (km)", value=20)
    family_size = st.selectbox("Family size", [1, 2, 3, 4], index=1)
    prefers_electric = st.checkbox("Prefer electric / hybrid")

    safe_budget, stretch_budget = compute_budget_eur(salary)

    candidates = filtered.copy()


    def score_car(row):
        score = 0

        # Affordability
        if row.price_eur <= safe_budget:
            score += 30
        elif row.price_eur <= stretch_budget:
            score += 15

        # Electric preference
        if prefers_electric and "elect" in str(row.fuel_type).lower():
            score += 25

        # Family preference
        if family_size > 2 and row.segment in ["suv", "kombi"]:
            score += 20

        # Newer = better
        score += max(0, row.year - 2015)

        # Lower mileage
        if pd.notna(row.mileage_km):
            score += max(0, 10 - (row.mileage_km / 50000))

        return score


    if len(candidates) > 0:
        candidates["score"] = candidates.apply(score_car, axis=1)
        top3 = candidates.sort_values("score", ascending=False).head(3)

        for _, row in top3.iterrows():
            st.markdown(
                f"### {row.make} {row.model} ({int(row.year)}) — €{int(row.price_eur):,}"
            )
            st.write(f"{int(row.mileage_km):,} km • {row.fuel_type} • {row.power_ps} PS")
            st.image(f"https://source.unsplash.com/400x200/?{row.make}+{row.model}", width=250)

    else:
        st.info("No cars match your filters.")


st.markdown("---")
st.caption("© AI German Car Advisor — Dataset local, costs estimated, not financial or insurance advice.")
