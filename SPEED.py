import streamlit as st
import speedtest
import pandas as pd
import plotly.graph_objects as go
import time
# The 'components' import is removed as it's not needed for local execution
# and was causing an error. We will use st.components.v1 directly.

def get_speedometer_html():
    """Returns the HTML/CSS/JS for the speedometer animation."""
    # This HTML/CSS/JS code is self-contained and creates the visual speedometer.
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        /* Basic styling for the body */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: transparent; /* Make background transparent */
        }
        /* Container for the gauge */
        .gauge-container {
            position: relative;
            width: 300px;
            height: 180px;
            text-align: center;
        }
        /* The background arc of the gauge */
        .gauge {
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 300px;
            height: 150px;
            border: 20px solid #e0e0e0;
            border-top: 20px solid #4CAF50; /* Green color for the top part */
            border-radius: 150px 150px 0 0;
            box-sizing: border-box;
            z-index: 1;
        }
        /* Inner part to create the hollow effect */
        .gauge-inner {
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 260px;
            height: 130px;
            background: white; /* Match Streamlit's typical background */
            border-radius: 130px 130px 0 0;
            z-index: 2;
        }
        /* The needle of the speedometer */
        .needle {
            position: absolute;
            bottom: 10px;
            left: 50%;
            width: 4px;
            height: 120px;
            background-color: #ff4b2b; /* A reddish-orange color */
            transform-origin: bottom center;
            transform: translateX(-50%) rotate(-90deg); /* Start at the far left */
            transition: transform 1.5s cubic-bezier(0.65, 0, 0.35, 1); /* Smooth transition */
            border-radius: 2px;
            z-index: 3;
            box-shadow: 0 0 5px rgba(0,0,0,0.5);
        }
        /* The center pivot point of the needle */
        .gauge-center {
            position: absolute;
            bottom: 0px;
            left: 50%;
            transform: translateX(-50%);
            width: 20px;
            height: 20px;
            background: #333;
            border-radius: 50%;
            z-index: 4;
        }
        /* Text to show the current status of the test */
        .status-text {
            position: absolute;
            bottom: 20px;
            width: 100%;
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            z-index: 5;
        }
        /* Styling for the number labels on the gauge */
        .gauge-label {
            position: absolute;
            color: #555;
            font-size: 0.9em;
            font-weight: bold;
            z-index: 2; /* Place labels above the inner gauge but below needle */
        }
        /* Positioning for each individual label */
        .label-0 { bottom: -5px; left: 25px; }
        .label-25 { top: 45px; left: 30px; }
        .label-50 { top: 10px; left: 50%; transform: translateX(-50%); }
        .label-75 { top: 45px; right: 30px; }
        .label-100 { bottom: -5px; right: 25px; }
    </style>
    </head>
    <body>

    <div class="gauge-container">
        <div class="gauge"></div>
        <div class="gauge-inner"></div>
        <div class="needle" id="needle"></div>
        <div class="gauge-center"></div>
        <div class="status-text" id="statusText">Initializing...</div>
        
        <!-- Number markings for the speedometer -->
        <div class="gauge-label label-0">0</div>
        <div class="gauge-label label-25">25</div>
        <div class="gauge-label label-50">50</div>
        <div class="gauge-label label-75">75</div>
        <div class="gauge-label label-100">100</div>
    </div>

    <script>
        const needle = document.getElementById('needle');
        const statusText = document.getElementById('statusText');

        // Function to rotate the needle. The gauge is a 180-degree arc.
        function setNeedleRotation(percent) {
            // Map a percentage (0-100) to a rotation angle (-90 to 90 degrees)
            const rotation = -90 + (percent * 1.8);
            needle.style.transform = `translateX(-50%) rotate(${rotation}deg)`;
        }

        // Simulates the needle moving during the test phases
        function animateTest() {
            statusText.textContent = 'Testing Download...';
            // Animate download phase
            setTimeout(() => setNeedleRotation(75), 100);
            setTimeout(() => setNeedleRotation(60), 2000);
            setTimeout(() => setNeedleRotation(85), 4000);

            // Animate upload phase
            setTimeout(() => {
                statusText.textContent = 'Testing Upload...';
                setNeedleRotation(0); // Reset for upload test
            }, 5500);
            setTimeout(() => setNeedleRotation(65), 5600);
            setTimeout(() => setNeedleRotation(50), 7500);
            setTimeout(() => setNeedleRotation(70), 9500);

            // Finalizing phase
            setTimeout(() => {
                statusText.textContent = 'Finalizing...';
                setNeedleRotation(0);
            }, 11000);
        }

        // Start the animation as soon as the component loads
        animateTest();
    </script>

    </body>
    </html>
    """
    return html_code

def run_speed_test():
    """Runs the speed test using the speedtest-cli library and returns the results."""
    try:
        s = speedtest.Speedtest()
        s.get_best_server()
        download_speed = s.download()
        upload_speed = s.upload()
        ping = s.results.ping

        return {
            "Download": download_speed / 1_000_000,  # Convert bits/s to Mbits/s
            "Upload": upload_speed / 1_000_000,    # Convert bits/s to Mbits/s
            "Ping": ping
        }
    except speedtest.SpeedtestException as e:
        st.error(f"A speed test error occurred: {e}")
        st.warning("Please check your internet connection and try again.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.warning("Please ensure 'speedtest-cli' is installed and your network is active.")
        return None

def create_gauge_chart(value, title, max_value, unit):
    """Creates a gauge chart for results using Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"{title} ({unit})"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2ca02c"}, # Green bar for the value
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [ # Color steps for the gauge background
                {'range': [0, max_value * 0.5], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [max_value * 0.8, max_value], 'color': 'rgba(0, 255, 0, 0.3)'}
            ],
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color="darkblue", family="Arial")
    )
    return fig

# --- Main Streamlit App Logic ---

st.set_page_config(page_title="Internet Speed Test", layout="centered", initial_sidebar_state="collapsed")

st.title("🌐 Internet Speed Tester")

st.write("""
This app uses `speedtest-cli` to measure your internet download speed, upload speed, and ping.
Click the button below to start the test.
""")

if st.button("🚀 Start Speed Test"):
    # Create a placeholder to hold the animation. This allows us to replace it later.
    animation_placeholder = st.empty()

    # The HTML component is now correctly placed inside the placeholder.
    with animation_placeholder.container():
        st.components.v1.html(get_speedometer_html(), height=220)

    # Run the actual speed test in the background while the animation is showing.
    results = run_speed_test()

    # Clear the animation from the placeholder.
    animation_placeholder.empty()

    # If the test was successful, display the results.
    if results:
        st.success("✅ Speed test completed!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.plotly_chart(create_gauge_chart(results['Download'], "Download", 200, "Mbps"), use_container_width=True)

        with col2:
            st.plotly_chart(create_gauge_chart(results['Upload'], "Upload", 100, "Mbps"), use_container_width=True)

        with col3:
            st.plotly_chart(create_gauge_chart(results['Ping'], "Ping", 100, "ms"), use_container_width=True)

        st.subheader("📊 Raw Results")
        st.metric("Download Speed", f"{results['Download']:.2f} Mbps")
        st.metric("Upload Speed", f"{results['Upload']:.2f} Mbps")
        st.metric("Ping", f"{results['Ping']:.2f} ms")

        df = pd.DataFrame([results])
        st.dataframe(df.style.format({
            "Download": "{:.2f}",
            "Upload": "{:.2f}",
            "Ping": "{:.2f}"
        }))
    else:
        # This message will show if run_speed_test returned None
        st.error("Could not retrieve speed test results.")

st.markdown("---")
st.info("Results may vary slightly depending on server load and network conditions.")
st.markdown("Made with ❤️ by JK")