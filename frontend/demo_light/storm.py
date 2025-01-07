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
from kamiwaza_client import KamiwazaClient


def get_kamiwaza_endpoint():
    """Get endpoint for the running Kamiwaza model deployment."""
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
    endpoint = f"http://{instance.host_name}:{deployment.lb_port}/v1/"
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
            try:
                if model_type == "Kamiwaza":
                    st.info("🤖 Using Kamiwaza's self-hosted model deployment")
                    api_base, model_name = get_kamiwaza_endpoint()
                    st.success(f"Connected to {model_name} at {api_base}")
                    
                    # Configure all LM instances to use Kamiwaza
                    from knowledge_storm.lm import KamiwazaModel
                    conv_simulator_lm = KamiwazaModel(model='model', max_tokens=500, api_base=api_base)
                    question_asker_lm = KamiwazaModel(model='model', max_tokens=500, api_base=api_base)
                    outline_gen_lm = KamiwazaModel(model='model', max_tokens=400, api_base=api_base)
                    article_gen_lm = KamiwazaModel(model='model', max_tokens=700, api_base=api_base)
                    article_polish_lm = KamiwazaModel(model='model', max_tokens=4000, api_base=api_base)

                    llm_configs.set_conv_simulator_lm(conv_simulator_lm)
                    llm_configs.set_question_asker_lm(question_asker_lm)
                    llm_configs.set_outline_gen_lm(outline_gen_lm)
                    llm_configs.set_article_gen_lm(article_gen_lm)
                    llm_configs.set_article_polish_lm(article_polish_lm)
                    
                    # Configure DSPy
                    import dspy
                    dspy.settings.configure(lm=conv_simulator_lm)
                else:
                    # Use existing OpenAI/Azure configuration
                    llm_configs.init_openai_model(
                        openai_api_key=st.secrets['OPENAI_API_KEY'],
                        azure_api_key=st.secrets.get('AZURE_API_KEY'),
                        openai_type=model_type.lower()
                    )
                    # Configure DSPy
                    import dspy
                    dspy.settings.configure(lm=llm_configs.conv_simulator_lm)

                # Configure retriever
                try:
                    from knowledge_storm.rm import SerperRM
                    rm = SerperRM(
                        serper_search_api_key=st.secrets['SERPER_API_KEY'],
                        query_params={'autocorrect': True, 'num': 10, 'page': 1}
                    )
                except KeyError:
                    st.error("Serper API key not configured. Please set SERPER_API_KEY in secrets.")
                    st.stop()

                # Create runner with configured LLMs
                runner = STORMWikiRunner(engine_args, llm_configs, rm)
                st.session_state["runner"] = runner

            except Exception as e:
                st.error(f"Error configuring model: {str(e)}")
                st.stop()

        if menu_selection == "My Articles":
            demo_util.clear_other_page_session_state(page_index=2)
            MyArticles.my_articles_page()
        elif menu_selection == "Create New Article":
            demo_util.clear_other_page_session_state(page_index=3)
            CreateNewArticle.create_new_article_page()


if __name__ == "__main__":
    main()
