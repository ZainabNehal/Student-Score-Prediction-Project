import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# Set page configuration
st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 Student Score Prediction Using Linear Regression")
st.markdown("""
This application predicts student exam scores based on study hours and attendance data using Linear Regression.
Upload your own CSV data or use the sample dataset to explore the model.
""")

@st.cache_data
def load_data(file_path):
    """Load and return the dataset"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def preprocess_data(data):
    """Clean and preprocess the data"""
    # Check for missing values
    if data.isnull().sum().any():
        st.warning("Missing values detected. Removing rows with missing data.")
        data = data.dropna()
    
    # Basic data validation
    if len(data) < 10:
        st.error("Dataset too small. Need at least 10 samples for reliable predictions.")
        return None
    
    return data

def create_visualizations(data):
    """Create exploratory data analysis visualizations"""
    st.subheader("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Hours studied distribution
        axes[0, 0].hist(data['Hours_Studied'], bins=15, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution of Study Hours')
        axes[0, 0].set_xlabel('Hours Studied')
        axes[0, 0].set_ylabel('Frequency')
        
        # Attendance distribution
        axes[0, 1].hist(data['Attendance'], bins=15, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Distribution of Attendance')
        axes[0, 1].set_xlabel('Attendance (%)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Final score distribution
        axes[1, 0].hist(data['Final_Score'], bins=15, alpha=0.7, color='salmon')
        axes[1, 0].set_title('Distribution of Final Scores')
        axes[1, 0].set_xlabel('Final Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # Remove empty subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Scatter plots
    st.subheader("🔍 Relationship Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data['Hours_Studied'], data['Final_Score'], alpha=0.6, color='blue')
        ax.set_xlabel('Hours Studied')
        ax.set_ylabel('Final Score')
        ax.set_title('Study Hours vs Final Score')
        # Add trend line
        z = np.polyfit(data['Hours_Studied'], data['Final_Score'], 1)
        p = np.poly1d(z)
        ax.plot(data['Hours_Studied'], p(data['Hours_Studied']), "r--", alpha=0.8)
        st.pyplot(fig)
        plt.close()
    
    with col4:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data['Attendance'], data['Final_Score'], alpha=0.6, color='green')
        ax.set_xlabel('Attendance (%)')
        ax.set_ylabel('Final Score')
        ax.set_title('Attendance vs Final Score')
        # Add trend line
        z = np.polyfit(data['Attendance'], data['Final_Score'], 1)
        p = np.poly1d(z)
        ax.plot(data['Attendance'], p(data['Attendance']), "r--", alpha=0.8)
        st.pyplot(fig)
        plt.close()

def train_model(data):
    """Train the Linear Regression model"""
    # Prepare features and target
    X = data[['Hours_Studied', 'Attendance']]
    y = data['Final_Score']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, scaler, X_test, y_test, y_pred, r2, mae, rmse

def display_model_performance(r2, mae, rmse, y_test, y_pred):
    """Display model performance metrics and visualizations"""
    st.subheader("📈 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
    with col2:
        st.metric("Mean Absolute Error", f"{mae:.2f}")
    with col3:
        st.metric("Root Mean Square Error", f"{rmse:.2f}")
    
    # Performance interpretation
    if r2 > 0.8:
        st.success("🎉 Excellent model performance!")
    elif r2 > 0.6:
        st.info("👍 Good model performance!")
    elif r2 > 0.4:
        st.warning("⚠️ Moderate model performance. Consider feature engineering.")
    else:
        st.error("❌ Poor model performance. More data or different features may be needed.")
    
    # Actual vs Predicted plot
    col4, col5 = st.columns(2)
    
    with col4:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Scores')
        ax.set_ylabel('Predicted Scores')
        ax.set_title('Actual vs Predicted Scores')
        st.pyplot(fig)
        plt.close()
    
    with col5:
        # Residuals plot
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Scores')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals Plot')
        st.pyplot(fig)
        plt.close()

def make_prediction(model, scaler, hours, attendance):
    """Make a prediction for new data"""
    # Prepare input data
    input_data = np.array([[hours, attendance]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

def main():
    # Sidebar for file upload
    st.sidebar.header("📁 Data Input")
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file", 
        type=['csv'],
        help="CSV should contain columns: Hours_Studied, Attendance, Final_Score"
    )
    
    # Use sample data option
    use_sample = st.sidebar.checkbox("Use sample dataset", value=True)
    
    data = None
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
    elif use_sample:
        # Load sample data
        if os.path.exists("student_data.csv"):
            data = load_data("student_data.csv")
        else:
            st.error("Sample dataset not found. Please upload your own CSV file.")
            return
    
    if data is not None:
        # Display data info
        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", data.shape)
            st.write("**Columns:**", list(data.columns))
        
        with col2:
            st.write("**First 5 rows:**")
            st.dataframe(data.head())
        
        # Check required columns
        required_columns = ['Hours_Studied', 'Attendance', 'Final_Score']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Missing required columns: {required_columns}")
            st.write("Your CSV should contain these exact column names.")
            return
        
        # Preprocess data
        processed_data = preprocess_data(data)
        
        if processed_data is not None:
            # Display statistics
            st.subheader("📊 Dataset Statistics")
            st.dataframe(processed_data.describe())
            
            # Create visualizations
            create_visualizations(processed_data)
            
            # Train model
            st.subheader("🤖 Model Training")
            with st.spinner("Training Linear Regression model..."):
                model, scaler, X_test, y_test, y_pred, r2, mae, rmse = train_model(processed_data)
            
            st.success("✅ Model trained successfully!")
            
            # Display performance
            display_model_performance(r2, mae, rmse, y_test, y_pred)
            
            # Feature importance
            st.subheader("🎯 Feature Importance")
            feature_names = ['Hours_Studied', 'Attendance']
            coefficients = model.coef_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Abs_Coefficient': np.abs(coefficients)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            col6, col7 = st.columns([1, 2])
            with col6:
                st.dataframe(importance_df)
            
            with col7:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(importance_df['Feature'], importance_df['Coefficient'])
                ax.set_xlabel('Coefficient Value')
                ax.set_title('Feature Coefficients')
                st.pyplot(fig)
                plt.close()
            
            # Prediction interface
            st.subheader("🔮 Make New Predictions")
            
            col8, col9, col10 = st.columns([1, 1, 2])
            
            with col8:
                hours_input = st.number_input(
                    "Study Hours",
                    min_value=0.0,
                    max_value=20.0,
                    value=4.0,
                    step=0.5,
                    help="Number of hours studied per week"
                )
            
            with col9:
                attendance_input = st.number_input(
                    "Attendance (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                    help="Attendance percentage"
                )
            
            with col10:
                if st.button("🎯 Predict Score", type="primary"):
                    predicted_score = make_prediction(model, scaler, hours_input, attendance_input)
                    
                    st.success(f"**Predicted Final Score: {predicted_score:.1f}**")
                    
                    # Add confidence interval estimate
                    std_error = rmse  # Use RMSE as approximation for standard error
                    confidence_interval = 1.96 * std_error  # 95% confidence interval
                    
                    st.info(f"**95% Confidence Interval: {predicted_score - confidence_interval:.1f} - {predicted_score + confidence_interval:.1f}**")
                    
                    # Interpretation
                    if predicted_score >= 90:
                        st.balloons()
                        st.success("🌟 Excellent predicted performance!")
                    elif predicted_score >= 80:
                        st.success("👍 Good predicted performance!")
                    elif predicted_score >= 70:
                        st.warning("⚠️ Average predicted performance. Consider increasing study time or attendance.")
                    else:
                        st.error("📚 Below average predicted performance. Significant improvement in study habits recommended.")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📝 Instructions:
    1. Upload a CSV file with columns: `Hours_Studied`, `Attendance`, `Final_Score`
    2. Or use the sample dataset
    3. Explore the data visualizations
    4. Review model performance metrics
    5. Make predictions for new students
    
    ### 📊 Example CSV format:
    ```
    Hours_Studied,Attendance,Final_Score
    5,90,85
    3,60,55
    6,95,90
    ```
    """)

if __name__ == "__main__":
    main()
