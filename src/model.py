"""Clustering and classification models for ERP student analytics."""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

CLUSTER_LABELS = {
    1: "High Achievers",
    2: "Moderate Performers",
    3: "Slow Learners",
    4: "Inconsistent Performers",
    5: "Potential Improvers",
}

RECOMMENDATIONS = {
    1: "Participate in advanced coursework and research projects; take leadership roles; seek mentorship and industry exposure; focus on networking and internships.",
    2: "Attend skill-development workshops and career counseling; improve time management; engage in peer learning; explore scholarships and financial aid where relevant.",
    3: "Consider remedial classes, academic counseling, tutoring, one-to-one mentorship, and structured study plans; use available wellbeing and financial-support resources.",
    4: "Identify causes of inconsistent performance; build consistent study habits; attend soft-skills and career programs; use personalized schedules and achievable goals.",
    5: "Use a structured academic improvement plan; participate in mentorship and skill-building; explore scholarship opportunities; engage in extracurricular activities.",
}

def run_kmeans(data, features, n_clusters=5):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    result = data.copy()
    result["cluster"] = kmeans.fit_predict(scaled) + 1
    result["cluster_label"] = result["cluster"].map(CLUSTER_LABELS)
    result["recommendations"] = result["cluster"].map(RECOMMENDATIONS)
    return result, scaler, kmeans

def evaluate_random_forest(data, features):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = data["cluster"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "report": classification_report(y_test, pred),
        "model": model,
    }
