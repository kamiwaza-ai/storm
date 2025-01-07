import os

script_dir = os.path.dirname(os.path.abspath(__file__))
wiki_root_dir = os.path.dirname(os.path.dirname(script_dir))

import demo_util
from pages_util import MyArticles, CreateNewArticle
import streamlit as st
from streamlit_float import *
from streamlit_option_menu import option_menu
from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner
from knowledge_storm.rm import YouRM


def get_kamiwaza_endpoint():
    """Get endpoint for the running Kamiwaza model deployment."""
    import os
    from kamiwaza_client import KamiwazaClient

    # Initialize Kamiwaza client
    client = KamiwazaClient(os.getenv("KAMIWAZA_API_URI"))
    
    # Get all deployments
    deployments = client.serving.list_deployments()
    
    # Find running deployments with running instances
    valid_deployments = []
    for deployment in deployments:
        if deployment.status == "DEPLOYED":
            running_instances = [i for i in deployment.instances if i.status == "DEPLOYED"]
            if running_instances:
                valid_deployments.append((deployment, running_instances[0]))
    
    if not valid_deployments:
        raise RuntimeError("No valid running Kamiwaza deployments found")
        
    # Sort by model size (using vram_allocation as proxy) and take largest
    deployment, instance = sorted(
        valid_deployments,
        key=lambda x: x[0].vram_allocation or 0,
        reverse=True
    )[0]
    
    # Construct endpoint
    endpoint = f"http://{instance.host_name}:{deployment.lb_port}/v1"
    return endpoint, deployment.m_name


def main():
    global database
    st.set_page_config(layout='wide')

    if "first_run" not in st.session_state:
        st.session_state['first_run'] = True

    # set api keys from secrets
    if st.session_state['first_run']:
        for key, value in st.secrets.items():
            if type(value) == str:
                os.environ[key] = value

    # initialize session_state
    if "selected_article_index" not in st.session_state:
        st.session_state["selected_article_index"] = 0
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = 0
    if st.session_state.get("rerun_requested", False):
        st.session_state["rerun_requested"] = False
        st.rerun()

    st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
    menu_container = st.container()
    with menu_container:
        pages = ["My Articles", "Create New Article"]
        styles={
                                         "container": {"padding": "0.2rem 0", 
                                                       "background-color": "#22222200"},
                                     }
        menu_selection = option_menu(None, pages,
                                     icons=['house', 'search'],
                                     menu_icon="cast", default_index=0, orientation="horizontal",
                                     manual_select=st.session_state.selected_page,
                                     styles=styles,
                                     key='menu_selection')
        if st.session_state.get("manual_selection_override", False):
            menu_selection = pages[st.session_state["selected_page"]]
            st.session_state["manual_selection_override"] = False
            st.session_state["selected_page"] = None

        # Configure STORM runner
        current_working_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")
        if not os.path.exists(current_working_dir):
            os.makedirs(current_working_dir)

        llm_configs = STORMWikiLMConfigs()
        engine_args = demo_util.get_engine_args()

        col1, col2 = st.columns([1, 3])
        with col1:
            model_type = st.selectbox(
                "Select Model",
                ["OpenAI", "Azure", "Kamiwaza"],
                index=0,
                help="Choose the model provider to use"
            )

        with col2:
            if model_type == "Kamiwaza":
                st.info("🤖 Using Kamiwaza private model deployment")
                try:
                    api_base, model_name = get_kamiwaza_endpoint()
                    st.success(f"Connected to {model_name} at {api_base}")
                    demo_util.configure_llm(llm_configs, model_type="Kamiwaza", api_base=api_base)
                except Exception as e:
                    st.error(f"Failed to connect to Kamiwaza: {str(e)}")
                    # Default to OpenAI if Kamiwaza fails
                    model_type = "OpenAI"
                    demo_util.configure_llm(llm_configs, model_type="OpenAI")
            else:
                demo_util.configure_llm(llm_configs, model_type=model_type)

        rm = YouRM(ydc_api_key=st.secrets['YDC_API_KEY'], k=engine_args.search_top_k)
        runner = STORMWikiRunner(engine_args, llm_configs, rm)
        st.session_state["runner"] = runner

        if menu_selection == "My Articles":
            demo_util.clear_other_page_session_state(page_index=2)
            MyArticles.my_articles_page()
        elif menu_selection == "Create New Article":
            demo_util.clear_other_page_session_state(page_index=3)
            CreateNewArticle.create_new_article_page()


if __name__ == "__main__":
    main()
