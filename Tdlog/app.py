import json
from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pickle
from collections import Counter

app = Flask(__name__)
app.secret_key = "secret-key"  # Nécessaire pour afficher les messages flash

# Chemins vers les fichiers du modèle et du scaler
model_path = "Projet_Tdlog_burn-out/Tdlog/model1.pkl"
scaler_path = "Projet_Tdlog_burn-out/Tdlog/scaler1.pkl"
data_file_path = "Projet_Tdlog_burn-out/employees_data.json"

# Charger le modèle
with  open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Charger le scaler
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Fonction pour charger les données des employés depuis un fichier
def load_employee_data():
    try:
        with open(data_file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []  # Si le fichier n'existe pas encore, retourner une liste vide

# Fonction pour sauvegarder les données des employés dans un fichier
def save_employee_data(data):
    with open(data_file_path, "w") as f:
        json.dump(data, f)

# Liste pour stocker les données des employés
employees_data = load_employee_data()
@app.route("/", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        role = request.form["role"]

        if role == "RH" and username == "RH" and password == "test":
            return redirect(url_for('rh'))
        elif role == "Employé" and username == "Employé" and password == "test":
            return redirect(url_for('employé'))
        else:
            flash("Nom d'utilisateur ou mot de passe incorrect", "danger")

    return render_template("signin.html")

# Route pour le formulaire d'employé
@app.route("/employé", methods=["GET", "POST"])
def employé():
    if request.method == "POST":
        
        # Récupérer les données du formulaire
        try:
            # Afficher les données reçues pour débogage
            print("Form Data:", request.form)

            nom = request.form["last-name"]
            prenom = request.form["first-name"]
            designation = float(request.form["designation"])
            resource_allocation = float(request.form["resource-allocation"])
            mental_fatigue_score = float(request.form["mental-fatigue"])

            gender = request.form["gender"]
            wfh_setup = request.form["wfh-setup"]

            # Préparation des données pour le modèle
            new_data = [[
                designation,
                resource_allocation,
                mental_fatigue_score,
                1 if gender == "Female" else 0,  # Gender_1
                1 if gender == "Male" else 0,    # Gender_0
                1 if wfh_setup == "Yes" else 0,  # WFH Setup Available_1
                1 if wfh_setup == "No" else 0   # WFH Setup Available_0
            ]]

            # Transformation des données et prédiction
            new_data_scaled = scaler.transform(np.array(new_data))
            burn_rate_prediction = model.predict(new_data_scaled)[0]

            # Calcul de la semaine réduite en fonction du burn-out rate
            if burn_rate_prediction <= 0.2:
                reduced_week = "Semaine 4"
            elif burn_rate_prediction <= 0.4:
                reduced_week = "Semaine 3"
            elif burn_rate_prediction <= 0.6:
                reduced_week = "Semaine 2"
            elif burn_rate_prediction <= 0.8:
                reduced_week = "Semaine 1"
            else:
                reduced_week = "Congé de 3 jours pour taux de burn-out trop élevé"

            # Enregistrement des données
            employees_data.append({
                "Nom": nom,
                "Prénom": prenom,
                "Burn-out Rate": round(burn_rate_prediction, 2),
                "Semaine réduite": reduced_week
            })

            # Sauvegarder les données
            save_employee_data(employees_data)

            # Message flash
            flash(f"Vos informations Monsieur/Madame: {prenom} {nom} ont été enregistrées avec succès. Votre responsable Rh vous communiquera votre semaine réduite en début du mois.", "success")
        except Exception as e:
            flash(f"Erreur lors du traitement des données : {str(e)}", "danger")

    return render_template("employé.html")

# Route pour afficher le tableau des employés
@app.route("/rh")
def rh():
    return render_template("rh.html", employees=employees_data)

@app.route("/profile")
def profile():
    return render_template("profile.html")


@app.route("/profiler")
def profiler():
    return render_template("profiler.html")


if __name__ == "__main__":
    app.run(debug=True)
