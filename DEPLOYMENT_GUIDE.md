# Streamlit Deployment Guide

This guide provides step-by-step instructions to deploy your **Liver Disease Analytics** project on Streamlit Cloud.

## Prerequisites

1.  A GitHub account with the repository `liver-disease-analytics` pushed.
2.  A [Streamlit Cloud](https://share.streamlit.io/) account (you can sign up using your GitHub account).

## Deployment Steps

1.  **Log in to Streamlit Cloud:**
    Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.

2.  **Create a New App:**
    Click the **"New app"** button in the top right corner.

3.  **Select Your Repository:**
    *   **Repository:** Select `Mmadrb/liver-disease-analytics` from the dropdown.
    *   **Branch:** Select `master`.
    *   **Main file path:** Enter `liver_disease.py`.

4.  **Advanced Settings (Optional):**
    If you need to specify a specific Python version or environment variables, click on "Advanced settings". For this project, the default settings should work fine.

5.  **Deploy:**
    Click the **"Deploy!"** button. Streamlit will now start building your app. This process includes:
    *   Setting up the environment.
    *   Installing dependencies from `requirements.txt`.
    *   Launching the Streamlit server.

6.  **Access Your App:**
    Once the deployment is complete, you will be provided with a URL (e.g., `https://liver-disease-analytics.streamlit.app/`) where your application is live.

## Troubleshooting

*   **Missing Dependencies:** If the app fails to start, check the logs in the Streamlit Cloud dashboard. Ensure all required packages are listed in `requirements.txt`.
*   **File Not Found:** Ensure `liver disease dataset.xlsx` is in the root directory of your repository, as the script expects it there.
*   **Python Version:** If you encounter issues with specific libraries, try specifying a different Python version in the Streamlit Cloud settings.

## Updating Your App

Whenever you push new changes to the `master` branch of your GitHub repository, Streamlit Cloud will automatically detect the changes and redeploy your application.
