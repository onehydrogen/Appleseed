# app.py
import pathlib
import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import pandas as pd
import logging
import sys
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
import tempfile
import requests
import traceback

# Set up deploy key if provided
if os.getenv('DEPLOY_KEY'):
    deploy_key_path = os.getenv('DEPLOY_KEY_PATH', '/app/deploy_key')
    try:
        os.makedirs(os.path.dirname(deploy_key_path), exist_ok=True)
        with open(deploy_key_path, 'w') as f:
            f.write(os.getenv('DEPLOY_KEY'))
        os.chmod(deploy_key_path, 0o600)
    except Exception as e:
        print(f"Error setting up deploy key: {e}")

# Load environment variables
load_dotenv()

# Constants
PAGE_SIZE = int(os.getenv('PAGE_SIZE', '10'))
DEPLOY_KEY_PATH = os.getenv('DEPLOY_KEY_PATH')
REPO_URL = os.getenv('REPO_URL', 'https://github.com/onehydrogen/Appleseed.git')
CSV_PATH = 'C:/Users/bendw/Downloads/legislative_analysis_AR_2025_20250217_111047 copy.csv'  # Update this line
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# Path configuration
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('legislative_dashboard_debug.log')
    ]
)
logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass


def format_hearing_date(hearing_str):
    """Formats hearing dates for consistent display."""
    try:
        if pd.isna(hearing_str) or hearing_str == 'N/A':
            return 'N/A'

        try:
            date_obj = datetime.strptime(hearing_str, '%Y-%m-%d %H:%M:%S')
            return date_obj.strftime('%b %d, %Y %I:%M %p')
        except:
            return hearing_str
    except Exception as e:
        logger.error(f"Error formatting hearing date: {e}")
        return hearing_str


def standardize_status(last_action):
    """Standardizes bill status values based on the last_action field."""
    if pd.isna(last_action):
        return 'pending'

    last_action = str(last_action).lower().strip()

    if any(term in last_action for term in [
        'is now act', 'became act', 'approved by the governor', 'signed by governor'
    ]):
        return 'passed'

    if any(term in last_action for term in [
        'died in house', 'died in senate', 'died in committee',
        'sine die', 'withdrawn', 'failed', 'vetoed'
    ]):
        return 'failed'

    return 'pending'


def get_sample_data():
    """Returns sample data when actual data is unavailable."""
    return pd.DataFrame({
        'year': ['2025', '2025'],
        'bill_number': ['HB1234', 'SB5678'],
        'title': ['Sample Bill 1', 'Sample Bill 2'],
        'primary_sponsors': ['John Doe', 'Jane Smith'],
        'party': ['D', 'R'],
        'district': ['District 1', 'District 2'],
        'status': ['pending', 'pending'],
        'last_action': ['Filed', 'In Committee'],
        'chamber': ['House', 'Senate'],
        'upcoming_hearings': ['N/A', 'N/A'],
        'past_hearings': ['N/A', 'N/A'],
        'state_bill_link': ['', '']
    })


def load_local_csv():
    """Loads CSV data from local data directory."""
    try:
        # Check for CSV files in the data directory
        csv_files = list(DATA_PATH.glob('*.csv'))

        if not csv_files:
            logger.warning("No CSV files found in data directory, using sample data")
            return get_sample_data()

        # Use the most recent CSV file if multiple exist
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading CSV from: {latest_csv}")

        df = pd.read_csv(latest_csv)
        if df.empty:
            logger.warning("Empty CSV file, using sample data")
            return get_sample_data()

        return parse_contents(df)

    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return get_sample_data()


def load_github_csv():
    """Loads CSV data with fallback options."""
    try:
        # First try to load from local path
        logger.info(f"Attempting to load CSV from local path: {CSV_PATH}")
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            if not df.empty:
                logger.info(f"Successfully loaded local CSV with {len(df)} rows")
                logger.info(f"Columns found: {df.columns.tolist()}")
                return parse_contents(df)

        # If local file doesn't exist or is empty, try GitHub
        logger.info("Local file not found or empty, trying GitHub...")
        import requests

        raw_url = "https://github.com/onehydrogen/Appleseed.git"
        logger.info(f"Attempting to fetch CSV from: {raw_url}")

        headers = {
            'Accept': 'text/csv',
            'User-Agent': 'Mozilla/5.0'
        }

        response = requests.get(raw_url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to fetch CSV from GitHub. Status code: {response.status_code}")
            logger.error(f"Response content: {response.text[:200]}")
            return get_sample_data()

        # Create a temporary file to store the CSV content
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', encoding='utf-8') as temp_file:
            temp_file.write(response.text)
            temp_file.flush()

            # Read the CSV using pandas
            df = pd.read_csv(temp_file.name)
            if df.empty:
                logger.warning("Empty CSV file from GitHub, using sample data")
                return get_sample_data()

            logger.info(f"Successfully loaded GitHub CSV with {len(df)} rows")
            logger.info(f"Columns found: {df.columns.tolist()}")
            return parse_contents(df)

    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        return get_sample_data()


def parse_contents(df):
    """Parses DataFrame and standardizes the data format."""
    try:
        logger.info(f"Processing DataFrame with columns: {df.columns.tolist()}")

        column_mapping = {
            'bill_number': 'bill_number',
            'title': 'title',
            'last_action': 'last_action',
            'sponsor_name': 'primary_sponsors',
            'sponsor_party': 'party',
            'sponsor_district': 'district',
            'session_year': 'year',
            'state_bill_link': 'state_bill_link',
            'upcoming_hearings': 'upcoming_hearings',
            'past_hearings': 'past_hearings'
        }

        df_processed = df.rename(columns=column_mapping)

        # Process hearing dates
        if 'upcoming_hearings' in df_processed.columns:
            df_processed['upcoming_hearings'] = df_processed['upcoming_hearings'].apply(format_hearing_date)
        if 'past_hearings' in df_processed.columns:
            df_processed['past_hearings'] = df_processed['past_hearings'].apply(format_hearing_date)

        # Extract co-sponsors
        if 'co_sponsors' not in df_processed.columns:
            if 'description' in df.columns:
                df_processed['co_sponsors'] = df['description'].apply(
                    lambda x: '; '.join(re.findall(r'Rep/. [A-Za-z/s]+(?:,|$)', str(x))) if pd.notna(x) else 'N/A'
                )
            else:
                df_processed['co_sponsors'] = 'N/A'

        # Determine chamber
        if 'chamber' not in df_processed.columns:
            df_processed['chamber'] = df_processed['bill_number'].apply(
                lambda x: 'House' if str(x).startswith('H') else 'Senate'
            )

        # Standardize status
        df_processed['status'] = df_processed['last_action'].apply(standardize_status)

        # Process links
        if 'state_bill_link' in df_processed.columns:
            df_processed['state_bill_link'] = df_processed['state_bill_link'].apply(
                lambda x: f"[View Bill]({x})" if pd.notna(x) else ""
            )
            df_processed['bill_number'] = df_processed.apply(
                lambda
                    row: f"[{row['bill_number']}]({row['state_bill_link'].replace('[View Bill]', '').replace('(', '').replace(')', '')})"
                if pd.notna(row['state_bill_link']) and row['state_bill_link'] != ""
                else row['bill_number'],
                axis=1
            )

        # Group by bill number
        grouped_df = df_processed.groupby('bill_number').agg({
            'year': 'first',
            'title': 'first',
            'status': 'last',
            'last_action': 'last',
            'primary_sponsors': 'first',
            'co_sponsors': 'first',
            'party': 'first',
            'district': 'first',
            'chamber': 'first',
            'state_bill_link': 'first',
            'upcoming_hearings': lambda x: '; '.join(filter(lambda v: v != 'N/A', x.unique())),
            'past_hearings': lambda x: '; '.join(filter(lambda v: v != 'N/A', x.unique()))
        }).reset_index()

        return grouped_df.fillna('N/A')

    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return get_sample_data()


def track_bill_progress(df):
    """Tracks bill progression through stages."""
    try:
        progress_stats = {
            'introduced': 0,
            'committee': 0,
            'floor_vote': 0,
            'passed_chamber': 0,
            'sent_to_governor': 0,
            'signed': 0
        }

        def determine_stage(last_action):
            if pd.isna(last_action):
                return 'introduced'

            last_action = str(last_action).lower()

            # Order matters here - check most specific conditions first
            if any(term in last_action for term in ['signed by governor', 'is now act', 'became act']):
                return 'signed'

            if any(term in last_action for term in ['to governor', 'transmitted to governor']):
                return 'sent_to_governor'

            if any(term in last_action for term in ['passed house', 'passed senate']):
                return 'passed_chamber'

            if any(term in last_action for term in ['third reading', 'floor vote', 'second reading']):
                return 'floor_vote'

            if any(term in last_action for term in ['referred to committee', 'in committee']):
                return 'committee'

            return 'introduced'

        for _, row in df.iterrows():
            stage = determine_stage(row['last_action'])
            progress_stats[stage] += 1

        total_bills = len(df)
        progress_percentages = {
            stage: (count / total_bills * 100) if total_bills > 0 else 0
            for stage, count in progress_stats.items()
        }

        return {
            'counts': progress_stats,
            'percentages': progress_percentages
        }

    except Exception as e:
        logger.error(f"Error in track_bill_progress: {e}")
        return None


def create_progress_tracker(progress_data):
    """Creates the progress tracker display component."""
    try:
        if not progress_data:
            return html.P("No progress data available", className="text-muted")

        return html.Div([
            html.H5("Bill Progress Breakdown", className="mb-3"),
            html.Div([
                dbc.Card(
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.P([
                                    html.Strong("Introduced: "),
                                    f"{progress_data['counts']['introduced']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['introduced']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ], className="mb-2"),
                                html.P([
                                    html.Strong("In Committee: "),
                                    f"{progress_data['counts']['committee']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['committee']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ], className="mb-2"),
                                html.P([
                                    html.Strong("Floor Vote: "),
                                    f"{progress_data['counts']['floor_vote']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['floor_vote']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ], className="mb-2"),
                                html.P([
                                    html.Strong("Passed Chamber: "),
                                    f"{progress_data['counts']['passed_chamber']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['passed_chamber']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ], className="mb-2"),
                                html.P([
                                    html.Strong("Sent to Governor: "),
                                    f"{progress_data['counts']['sent_to_governor']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['sent_to_governor']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ], className="mb-2"),
                                html.P([
                                    html.Strong("Signed into Law: "),
                                    f"{progress_data['counts']['signed']} bills ",
                                    html.Span(
                                        f"({progress_data['percentages']['signed']:.1f}%)",
                                        className="percentage-badge"
                                    )
                                ], className="mb-2"),
                            ], className="pl-4 border-l-4 border-blue-200")
                        ])
                    ]),
                    className="shadow-sm mb-3"
                )
            ])
        ])
    except Exception as e:
        logger.error(f"Error creating progress tracker: {e}")
        return html.P("Error displaying progress data", className="text-danger")


def create_hearings_card(df, search_performed=False):
    """Creates a card displaying latest hearing information."""
    try:
        if not search_performed:
            return html.Div([
                html.H5("Hearing Schedule", className="mb-3"),
                dbc.Card(
                    dbc.CardBody([
                        html.P("Use search bar to display latest hearing info",
                              className="text-muted text-center")
                    ]),
                    className="shadow-sm"
                )
            ])

        if 'latest_hearing_data' not in df.columns:
            return html.P("Latest hearing information not available", className="text-muted")

        latest_hearings = df[df['latest_hearing_data'].notna() & (df['latest_hearing_data'] != 'N/A')]

        return html.Div([
            html.H5("Latest Hearing Information", className="mb-3"),
            dbc.Card(
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.P([
                                html.Strong(row['bill_number']), ": ",
                                row['latest_hearing_data']
                            ], className="mb-2")
                            for _, row in latest_hearings.iterrows()
                        ]) if not latest_hearings.empty else html.P("No hearing information available")
                    ])
                ]),
                className="shadow-sm"
            )
        ])
    except Exception as e:
        logger.error(f"Error creating hearings card: {e}")
        return html.P("Error loading hearing information", className="text-danger")


def calculate_sponsor_stats(df, search_value):
    """Calculates comprehensive sponsorship statistics."""
    try:
        bill_outcomes = defaultdict(int)
        primary_bills = df[df['primary_sponsors'].str.contains(search_value, case=False, na=False)]

        num_primary = len(primary_bills)
        total_bills = len(df)
        primary_percentage = (num_primary / total_bills * 100) if total_bills > 0 else 0

        for _, row in primary_bills.iterrows():
            status = standardize_status(row['last_action'])
            bill_outcomes[status] += 1

        completed_bills = bill_outcomes['passed'] + bill_outcomes['failed']
        bill_outcomes['total'] = num_primary
        bill_outcomes['success_rate'] = (
            (bill_outcomes['passed'] / completed_bills * 100)
            if completed_bills > 0 else 0
        )

        progress_analysis = track_bill_progress(primary_bills)

        return {
            'primary_bills': num_primary,
            'primary_percentage': primary_percentage,
            'bill_outcomes': bill_outcomes,
            'progress_analysis': progress_analysis
        }

    except Exception as e:
        logger.error(f"Error calculating sponsor stats: {e}")
        return None


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
    ]
)
server = app.server

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Legislative Analytics Dashboard</title>
        {%css%}
        <style>
            .nav-link:hover { background-color: #e9ecef; border-radius: 5px; }
            .card { 
                transition: transform 0.2s;
                border-left: 4px solid #D7B547; /* Gold color */
            }
            .card:hover { transform: translateY(-5px); }
            .dashboard-title { 
                background: linear-gradient(120deg, #580100, #D7B547); /* Red and Gold gradient */
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .percentage-badge {
                background-color: #580100; /* Red color */
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 9999px;
                font-size: 0.875rem;
            }
            .navbar { background-color: #580100 !important; } /* Red color for navbar */
            .btn-primary { background-color: #D7B547; border-color: #D7B547; } /* Gold color for primary buttons */
            .btn-primary:hover { background-color: #C0A03D; border-color: #C0A03D; } /* Darker gold on hover */
            .text-primary { color: #580100 !important; } /* Red color for primary text */
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Navigation bar with logo
# Navigation bar with logo
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Donate", href="https://www.arappleseed.org/donate", active=True)),
        dbc.NavItem(dbc.NavLink("About", href="https://www.arappleseed.org/", target="_blank")),
    ],
    brand=html.Img(src="assets/AR_Appleseed_logo.png",height="40px"),  # Updated to .png
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

# Search section
search_section = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Input(
                    id="search-bar",
                    placeholder="Search Bills, Sponsors, or Topics...",
                    type="text",
                    className="mb-3"
                ),
            ], width=8),
            dbc.Col([
                dbc.Button("Search", id="search-button", color="primary", className="mr-2"),
                dbc.Button("Clear", id="clear-search", color="secondary")
            ], width=4)
        ])
    ]),
    className="mb-4"
)


def create_stat_card(title, icon, content):
    return dbc.Card(
        dbc.CardBody([
            html.H4([html.I(className=f"fas {icon} mr-2"), title],
                    className="text-primary"),
            html.Hr(),
            html.Div(id=content, className="mt-3")
        ]),
        className="mb-4 shadow-sm"
    )


# Main layout
app.layout = html.Div([
    dcc.Store(id='original-data'),
    dcc.Location(id='url', refresh=False),

    # Error Alert
    dbc.Alert(
        id='error-message',
        is_open=False,
        duration=5000,
        className="mb-3"
    ),

    navbar,
    dbc.Container([
        # Title section
        html.Div([
            html.H1("Arkansas Appleseed Legal Justice Center", className="mb-0"),
            html.H1(" Legislative Tracker", className="mb-0"),
            html.P("Track, Analyze, and Understand Legislative Data",
                   className="lead mb-0")
        ], className="dashboard-title text-center mb-4"),

        search_section,

        # Stats cards
        dbc.Row([
            dbc.Col(create_stat_card(
                "Sponsorship Overview",
                "fa-users",
                "sponsorship-stats"
            ), md=3),
            dbc.Col(create_stat_card(
                "Bill Outcomes",
                "fa-chart-pie",
                "bill-outcomes"
            ), md=3),
            dbc.Col(create_stat_card(
                "Bill Progress",
                "fa-tasks",
                "progress-tracker"
            ), md=3),
            dbc.Col(create_stat_card(
                "Hearing Schedule",
                "fa-calendar",
                "hearing-schedule"
            ), md=3),
        ]),

        # Bills table
        dbc.Card(
            dbc.CardBody([
                html.H3("Bills Overview", className="mb-4"),
                dash_table.DataTable(
                    id="bills-table",
                    columns=[
                        {"name": "Year", "id": "year"},
                        {"name": "Bill Number", "id": "bill_number", "presentation": "markdown"},
                        {"name": "Title", "id": "title"},
                        {"name": "Primary Sponsor", "id": "primary_sponsors"},
                        {"name": "Party", "id": "party"},
                        {"name": "District", "id": "district"},
                        {"name": "Status", "id": "status"},
                        {"name": "Last Action", "id": "last_action"},
                        {"name": "Chamber", "id": "chamber"},
                        {"name": "Upcoming Hearing", "id": "upcoming_hearings"},
                        {"name": "Past Hearings", "id": "past_hearings"},
                        {"name": "Link", "id": "state_bill_link", "presentation": "markdown"}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '15px',
                        'fontFamily': '"Segoe UI", sans-serif'
                    },
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold',
                        'border': '1px solid #dee2e6'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        }
                    ],
                    page_size=PAGE_SIZE,
                    page_current=0,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    row_selectable="single",
                    selected_rows=[]
                )
            ]),
            className="mb-4"
        ),

        # Footer
        html.Footer(
            dbc.Row([
                dbc.Col(
                    html.P("© 2024 Legislative Analytics. All rights reserved.",
                           className="text-muted"),
                    className="text-center mt-4"
                )
            ])
        )
    ], fluid=True)
])


# Callbacks
@app.callback(
    [Output("bills-table", "data"),
     Output('sponsorship-stats', 'children'),
     Output('bill-outcomes', 'children'),
     Output('progress-tracker', 'children'),
     Output('hearing-schedule', 'children'),
     Output('original-data', 'data'),
     Output('error-message', 'children'),
     Output('error-message', 'is_open'),
     Output('error-message', 'color')],
    [Input('search-button', 'n_clicks'),
     Input('clear-search', 'n_clicks'),
     Input('url', 'pathname')],
    [State("search-bar", "value"),
     State("original-data", "data"),
     State("bills-table", "data")],
    prevent_initial_call=False
)
def update_dashboard(search_clicks, clear_clicks, pathname, search_value, original_data, current_data):
    """Callback to update dashboard components with error handling."""
    triggered_id = ctx.triggered_id if ctx.triggered_id is not None else 'url'

    try:
        # Load initial data if none exists
        if original_data is None:
            # Load data directly from GitHub
            df = load_github_csv()
            if df.empty:
                return [], "No data", "No data", "No data", "No data", None, "No data available", True, "warning"
            # Create initial overview statistics
            overview_stats = html.Div([
                html.P(f"Total Bills: {len(df)}", className="stat-item"),
                html.P(f"Unique Sponsors: {df['primary_sponsors'].nunique()}", className="stat-item")
            ])

            status_counts = df['status'].value_counts()
            bill_outcomes = html.Div([
                html.P(f"Passed: {status_counts.get('passed', 0)}", style={"color": "#2f855a"}),
                html.P(f"Failed: {status_counts.get('failed', 0)}", style={"color": "#c53030"}),
                html.P(f"Pending: {status_counts.get('pending', 0)}", style={"color": "#744210"})
            ])

            progress_data = track_bill_progress(df)
            progress_tracker = create_progress_tracker(progress_data)
            hearing_schedule = create_hearings_card(df)

            return (df.to_dict('records'), overview_stats, bill_outcomes, progress_tracker,
                    hearing_schedule, df.to_dict('records'), None, False, "success")

        # Handle search functionality
        if triggered_id == 'search-button' and search_value:
            df = pd.DataFrame(original_data)
            filtered_df = df[
                df['bill_number'].str.contains(search_value, case=False, na=False) |
                df['title'].str.contains(search_value, case=False, na=False) |
                df['primary_sponsors'].str.contains(search_value, case=False, na=False)
                ]

            if filtered_df.empty:
                return ([], "No results", "No results", "No results", "No results",
                        original_data, f"No results found for '{search_value}'", True, "warning")

            sponsor_stats = calculate_sponsor_stats(df, search_value)
            if sponsor_stats:
                stats_display = html.Div([
                    html.P(f"Primary Bills: {sponsor_stats['primary_bills']}", className="stat-item"),
                    html.P(f"Success Rate: {sponsor_stats['bill_outcomes']['success_rate']:.1f}%",
                           className="stat-item")
                ])

                outcomes = html.Div([
                    html.P(f"Passed: {sponsor_stats['bill_outcomes']['passed']}", style={"color": "#2f855a"}),
                    html.P(f"Failed: {sponsor_stats['bill_outcomes']['failed']}", style={"color": "#c53030"}),
                    html.P(f"Pending: {sponsor_stats['bill_outcomes'].get('pending', 0)}",
                           style={"color": "#744210"})
                ])

                progress = create_progress_tracker(sponsor_stats['progress_analysis'])
                hearings = create_hearings_card(filtered_df)

                return (filtered_df.to_dict('records'), stats_display, outcomes, progress,
                        hearings, original_data, None, False, "success")

        # Handle clear functionality
        elif triggered_id == 'clear-search':
            if original_data:
                df = pd.DataFrame(original_data)
                overview_stats = html.Div([
                    html.P(f"Total Bills: {len(df)}", className="stat-item"),
                    html.P(f"Unique Sponsors: {df['primary_sponsors'].nunique()}", className="stat-item")
                ])

                status_counts = df['status'].value_counts()
                bill_outcomes = html.Div([
                    html.P(f"Passed: {status_counts.get('passed', 0)}", style={"color": "#2f855a"}),
                    html.P(f"Failed: {status_counts.get('failed', 0)}", style={"color": "#c53030"}),
                    html.P(f"Pending: {status_counts.get('pending', 0)}", style={"color": "#744210"})
                ])

                progress_data = track_bill_progress(df)
                progress_tracker = create_progress_tracker(progress_data)
                hearing_schedule = create_hearings_card(filtered_df, search_performed=True)

                return (df.to_dict('records'), overview_stats, bill_outcomes, progress_tracker,
                        hearing_schedule, original_data, None, False, "success")

        return current_data or [], "No data", "No data", "No data", "No data", original_data, None, False, "success"

    except Exception as e:
        logger.error(f"Error in dashboard update: {e}")
        return (current_data or [], "Error", "Error", "Error", "Error", original_data,
                f"An error occurred: {str(e)}", True, "danger")


if __name__ == '__main__':
    app.run_server(debug=DEBUG_MODE, port=8051)  # or any other port number