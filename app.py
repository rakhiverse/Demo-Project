# app.py

import os
import gradio as gr
import joblib

# Load the trained model at startup
deployed_model = joblib.load('loan_approval_model.pkl')


def predict_loan(no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score):
    # Encode categorical inputs (same mapping used during training)
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0

    # The model expects a 2D array matching the exact order of x_train
    input_data = [[
        no_of_dependents,
        education_val,
        self_employed_val,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
    ]]
    prediction = deployed_model.predict(input_data)

    # Interpret the binary outcome
    if prediction[0] == 1:
        return "Prediction: Loan Approved"
    else:
        return "Prediction: Loan Rejected"

interface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Radio(["Graduate", "Not Graduate"], label="Education"),
        gr.Radio(["Yes", "No"], label="Self Employed"),
        gr.Number(label="Annual Income (INR)"),
        gr.Number(label="Loan Amount (INR)"),
        gr.Number(label="Loan Term (Years)"),
        gr.Number(label="CIBIL Score"),
    ],
    outputs=gr.Text(label="Assessment Result"),
    title="Loan Approval Prediction System",
    description="Enter the applicant details to predict loan approval using a Logistic Regression Machine Learning model.",
    article="<center><h1>Created by Software Developer</h1></center>",
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7861)),
        footer_links=[],
    )

