def format_probability(probability):
    return f"{probability * 100:.2f}%"


def get_risk_color(risk_level):
    if risk_level == "High Risk":
        return "red"

    if risk_level == "Medium Risk":
        return "orange"

    return "green"


def generate_retention_recommendation(probability, cltv):
    if probability >= 0.70 and cltv >= 4000:
        return (
            "This is a high-value customer with high churn risk. "
            "Prioritize immediate retention outreach, offer a loyalty discount, "
            "and assign proactive customer support."
        )

    if probability >= 0.70:
        return (
            "This customer has high churn risk. "
            "Offer a personalized retention plan, review service issues, "
            "and encourage a longer-term contract."
        )

    if probability >= 0.40 and cltv >= 4000:
        return (
            "This is a valuable customer with moderate churn risk. "
            "Monitor engagement closely and offer targeted benefits before risk increases."
        )

    if probability >= 0.40:
        return (
            "This customer has moderate churn risk. "
            "Improve engagement through service check-ins, offers, or plan optimization."
        )

    return (
        "This customer currently has low churn risk. "
        "Continue regular engagement and maintain service quality."
    )


def explain_risk_level(risk_level):
    if risk_level == "High Risk":
        return (
            "The customer has a strong likelihood of churn. "
            "Retention action should be taken soon."
        )

    if risk_level == "Medium Risk":
        return (
            "The customer shows some churn signals. "
            "Preventive engagement is recommended."
        )

    return (
        "The customer is currently unlikely to churn. "
        "Standard engagement is sufficient."
    )