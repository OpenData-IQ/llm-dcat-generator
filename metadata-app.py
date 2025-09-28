import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from jinja2 import Template
import re
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


st.set_page_config(
    page_title="Automatisierte Generierung von Metadatenbeschreibungen",
    layout="wide",
)


prompt3_description = """
Du bist professioneller Redakteur für Open-Data-Kataloge. Deine Aufgabe ist es, präzise, gut strukturierte und verständliche Beschreibungen für Datensätze zu generieren.

Schreibe Sätze nach diesem Muster:
1) Grundbasis
2) Ableitung
3) Details
4) Varianten/Aktualisierung
Nutze keine anderen Satzfolgen.

- Paraphrasiere eng am Originalstil, ohne neue Fakten hinzuzufügen.
- Lasse alle Prompt-Labels („Titel:“, „Beschreibung:“, „#…#“) in der Ausgabe weg
- Der Ziel-Datensatz stammt aus dem Bundesland Sachsen-Anhalt.
- Beziehe dich nie auf andere Bundesländer.
"""

sparql_k_thresholds = [5, 4, 3, 2, 1, 0]


def load_sparql(path: str = "data/1template.sparql") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            "'template.sparql' nicht gefunden. Lege die Datei in denselben Ordner wie die App."
        )

def render_sparql(template_text: str, *, dataset_url: str, count_threshold: int, limit: int) -> str:
    template = Template(template_text)
    return template.render(dataset=dataset_url, count_threshold=int(count_threshold), limit=int(limit))


@dataclass
class ParameterPr:
    prompt_type: str
    temperature: float
    frequency_penalty: float
    presence_penalty: float
    max_tokens: int
    num_shots: int


def prompt_states() -> None:
    st.session_state.setdefault("last_run", None)
    st.session_state.setdefault("df", pd.DataFrame())
    st.session_state.setdefault("last_generated_description", "")
    st.session_state.setdefault("last_title_keywords", "")
    st.session_state.setdefault("param_dialog_open", False)
    st.session_state.setdefault("proceed", False)
    st.session_state.setdefault("auto_run_desc", False)
    st.session_state.setdefault("last_generated_keywords", [])
    st.session_state.setdefault("last_generated_rdf", "")
    st.session_state.setdefault("last_fewshot_titles", [])
    st.session_state.setdefault("last_title_description", "")
    st.session_state.setdefault("auto_keywords", False)

    if "sparql_template_text" not in st.session_state:
        try:
            st.session_state["sparql_template_text"] = load_sparql()
        except Exception:
            st.session_state["sparql_template_text"] = ""

def css_load(path: str = "data/1style.css") -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def load_csv(source_path: Optional[str], uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif source_path:
        df = pd.read_csv(source_path)
    else:
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.lower()
    required = {"title", "description", "dataset"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Pflichtspalten fehlen: {missing}")
    

    df["title"] = df["title"].astype(str).str.strip()
    df["description"] = df["description"].astype(str)
    df["dataset"] = df["dataset"].astype(str).str.strip()
    return df

def parameter_warning(p: ParameterPr) -> tuple[bool, str]:
    reasons: list[str] = []

    if p.temperature > 1.2:
        reasons.append(
            f"(Temperatur {(p.temperature)} > {1.2} kann inkonsistente und fehlerhafte Formulierungen verursachen."
        )

    if p.max_tokens < 300:
        reasons.append(
            f"Max Tokens {p.max_tokens} < {300} kann Texte mitte im Satz abbrechen."
        )

    if not (-1.0 <= p.frequency_penalty <= 1.0):
        reasons.append(
            f"Frequency Penalty {p.frequency_penalty} (außerhalb −1.0 … +1.0) kann eine unnatürlich starke Wiederholungsvermeidung verursachen."
        )

    if not (-1.0 <= p.presence_penalty <= 1.0):
        reasons.append(
            f"Presence Penalty {p.presence_penalty} (außerhalb −1.0 … +1.0) kann zu stärkeren Themenabweichungen führen."
        )

    if not reasons:
        return False, ""

    achtung_msg = (
        "Achtung: Ihre aktuelle Parametereinstellung kann die Qualität der Generierung negativ beeinflussen.\n\n"
        + "\n".join(reasons)
        + "\n\nMöchten Sie die Parameter anpassen oder fortfahren?"
    )
    return True, achtung_msg


dialog = getattr(st, "dialog", getattr(st, "experimental_dialog"))


@dialog("ACHTUNG")
def parameterModal():
    achtung_msg = st.session_state.get("param_dialog_message", "")
    action = st.session_state.get("param_dialog_action", "")

    st.warning(achtung_msg)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Parameter ändern", key=f"dlg_change_{action}"):
            st.session_state["param_dialog_open"] = False
            st.session_state["proceed"] = False
            st.session_state.pop("param_dialog_action", None)
            st.session_state.pop("param_dialog_message", None)
            st.rerun()
    with c2:
        if st.button("Trotzdem fortfahren", key=f"dlg_proceed_{action}"):
            st.session_state["param_dialog_open"] = False
            st.session_state["proceed"] = True
            if action == "desc":
                st.session_state["auto_run_desc"] = True
            elif action == "kw":
                st.session_state["auto_keywords"] = True
            st.session_state.pop("param_dialog_action", None)
            st.session_state.pop("param_dialog_message", None)
            st.rerun()




def get_fewshots(
    dataset_url: str,
    thresholds: List[int],
    limit: int,
    template_text_override: Optional[str] = None, 
) -> tuple[list[dict], int, str]:

    template_text = template_text_override if template_text_override is not None and template_text_override.strip() != "" else load_sparql()

    last_rendered = ""
    for k in thresholds:
        rendered = render_sparql(
            template_text,
            dataset_url=f"<{dataset_url}>",
            count_threshold=int(k),
            limit=int(limit),
        )
        last_rendered = rendered

        sparql = SPARQLWrapper("https://www.govdata.de/sparql")
        sparql.setQuery(rendered)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
        except Exception as e:
            raise RuntimeError(f"SPARQL-Fehler: {e}")

        examples: list[dict] = []
        for b in bindings:
            title = b.get("title", {}).get("value", "").strip()
            description = b.get("description", {}).get("value", "").strip()
            if title and description:
                examples.append({"title": title, "description": description})

        if len(examples) >= limit or k == thresholds[-1]:
            return examples[:limit], k, rendered

    return [], thresholds[-1], last_rendered


def get_openai_api_key() -> Optional[str]:
    load_dotenv()  
    key = os.getenv("OPENAI_API_KEY")
    return key


def build_fewshot_messages(
    *,
    system_prompt_text: str,
    fewshots: list[dict],
    dataset_title: str,
) -> tuple[list, str]:


    examples = []
    for ex in (fewshots or []):
        t = (ex.get("title") or "").strip()
        d = (ex.get("description") or "").strip()
        if t and d:
            examples.append({"title": t, "description": d})

    example_prompt = PromptTemplate.from_template(
        "Beispiel"
        "\nTitel: {title}"
        "\nBeschreibung: {description}\n"
    )

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix=(
            "Jetzt ein neuer Datensatz\n"
            "Titel: {input}\n"
            "Beschreibung:"
        ),
        input_variables=["input"],
    )

    prompt_string = prompt.invoke({"input": dataset_title}).to_string()

    messages = [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=prompt_string),
    ]
    return messages, prompt_string


def runLLM_messages(
    messages: List,
    *,
    temperature: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    model_name: str = "gpt-4o-mini",
) -> str:
    api_key = get_openai_api_key()

    llm = ChatOpenAI(
        api_key = api_key,
        model = model_name,
        temperature = temperature,
        frequency_penalty= frequency_penalty,
        presence_penalty = presence_penalty,
        max_tokens = max_tokens,
    )
    try:
        out = llm.invoke(messages).content
        return out.strip()
    except Exception as e:
        raise RuntimeError(f"LLM-Fehler: {e}")



def keywords_prompt(*, title: str) -> list:
    system = SystemMessage(
        content=(
            """
 Du bist professioneller Redakteur für Open-Data-Kataloge und generierst präzise dcat:keywords (Schlagwörter) für Open-Data-Datensätze unter folgenden Regeln
 - Generiere Genau 10 Begriffe auf deutsch
 - Keine Satzzeichen, Ziffern oder Sonderzeichen
 - Maximal 1 Wort pro Schlagwort 
 - "sachsen anhalt" kann zusammen geschrieben werden mit Leerzeichen
 - Keine Bindestriche "-"
 - Schlagwörter klein schreiben
 - Ausgabe als JSON-Array aus 10 Strings
 - Bitte keinen zusätzlichen Text vor oder nach dem JSON
 - Übernheme keinesfalls den exakten Titel
 - Alle Daten sind dem Bundesland Sachsen-Anhalt zu zuordnen
 - Vermeide zu allgemeine Schlagwörter
"""
        )
    )
    human = HumanMessage(
        content=(f"Titel: {title}\n\n" "Erzeuge jetzt  10 dcat:keywords")
    )
    return [system, human]


def keywordsFormat(raw: str) -> list[str]:
    try:
        data = json.loads(raw)

        cleaned: list[str] = []
        seen = set()
        for item in data:
            if not isinstance(item, str):
                continue
            kw = item.strip()
            kw = kw.strip(" ,;:\"'").lower()
            if not kw or len(kw.split()) > 3:
                continue
            if kw not in seen:
                seen.add(kw)
                cleaned.append(kw)

        if len(cleaned) != 10:
            raise ValueError(f"Es wurden nur {len(cleaned)} Schalgwörter erkannt, erwartet: 10.")
        return cleaned
    except Exception as e:
        raise RuntimeError(f"Keywords konnten nicht zuverlässig geparst/validiert werden: {e}")


def generate_keywordsLLM(
    *,
    title: str,
    model_name: str,
    temperature: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 256,
) -> list[str]:
    messages = keywords_prompt(title=title)
    raw = runLLM_messages(
        messages,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        model_name=model_name,
    )
    return keywordsFormat(raw)



def _xml_escape(s: str) -> str:
    if s is None:
        return ""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def buildRDF(
    *,
    dataset_uri: str,
    title: str,
    description: str,
    keywords: list[str] | None,
    lang: str = "de",
) -> str:
    ds_url = _xml_escape(dataset_uri.strip()) if dataset_uri else "___"
    _title = _xml_escape(title.strip()) if title else "___"
    _desc = _xml_escape(description.strip()) if description else "____"

    kw_lines = []
    if keywords:
        for kw in keywords:
            kw_norm = _xml_escape((kw or "").strip())
            if kw_norm:
                kw_lines.append(f'  <dcat:keyword xml:lang="{lang}">{kw_norm}</dcat:keyword>')

    rdf_gerüst = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<rdf:RDF",
        ' xmlns:vcard="http://www.w3.org/2006/vcard/ns#"',
        ' xmlns:dcat="http://www.w3.org/ns/dcat#"',
        ' xmlns:dct="http://purl.org/dc/terms/"',
        ' xmlns:foaf="http://xmlns.com/foaf/0.1/"',
        ' xmlns:dcatde="http://dcat-ap.de/def/dcatde/"',
        ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        "",
        f'  <dcat:Dataset rdf:about="{ds_url}">',
        f'    <dct:title xml:lang="{lang}">{_title}</dct:title>',
        f'    <dct:description xml:lang="{lang}">{_desc}</dct:description>',
    ]
    if kw_lines:
        rdf_gerüst.extend(kw_lines)
    rdf_gerüst.extend(
        [
            "",
            "    <dct:publisher>",
            "      <foaf:Agent>",
            "        <foaf:name>___</foaf:name>",
            "        <foaf:mbox>___</foaf:mbox>",
            '        <foaf:homepage rdf:resource="___"/>',
            "      </foaf:Agent>",
            "    </dct:publisher>",
            "",
            "    <dcat:contactPoint>",
            "      <vcard:Organization>",
            "        <vcard:fn>___</vcard:fn>",
            '        <vcard:hasEmail rdf:resource="___"/>',
            '        <vcard:hasURL rdf:resource="___"/>',
            "      </vcard:Organization>",
            "    </dcat:contactPoint>",
            "",
            '    <dcatde:contributorID rdf:resource="___"/>',
            "",
            "    <dcat:distribution>",
            '      <dcat:Distribution rdf:about="___">',
            '        <dcat:accessURL rdf:resource="___"/>',
            '        <dct:license rdf:resource="___"/>',
            '        <dct:license rdf:resource="___"/>',
            '        <dct:format rdf:resource="___"/>',
            "      </dcat:Distribution>",
            "    </dcat:distribution>",
            "",
            "  </dcat:Dataset>",
            "</rdf:RDF>",
        ]
    )
    return "\n".join(rdf_gerüst)


def main() -> None:
    prompt_states()
    css_load()

    # HTML Header 
    st.markdown(
        """
 <div class="Header-Container">
   <div class="Header">
     <div class="Header-Icon"></div>
     <div class="Header-Text">
       <h1>Metadaten für Open-Data automatisiert erstellen </h1>
       <p class="beschreibung">Beschreibungen, Schlagwörter und RDF (DCAT) mithilfe von Künstlicher Intelligenz erstellen.</p>
     </div>
   </div>
 </div>
        """,
        unsafe_allow_html=True,
    )

#ST UI

    # 1 Upload/Eingabe
    #col_left, col_right = st.columns([2, 1], gap="small", border=True)
    col_left, col_right = st.columns([2, 1], gap="small")
    with col_left:
        with st.container():
            st.markdown("<h2 style='color: #1e5fcfff;'>1. Datensätze hinzufügen</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p>Laden Sie eine CSV-Datei per Drag and Drop hoch oder geben Sie die Pflichtfelder Titel und Beschreibung manuell ein.</p>",
                unsafe_allow_html=True,
            )

            tab_csv, tab_manual = st.tabs(["CSV hochladen", "Manuelle Eingabe"])

            df: pd.DataFrame = st.session_state.get("df")

            with tab_csv:
                up_file = st.file_uploader(" ", type=["csv"], accept_multiple_files=False)
                if st.button("↗ CSV hochladen", type="primary", use_container_width=False):
                    try:
                        if up_file is None:
                            st.warning("Bitte zuerst eine CSV-Datei auswählen.")
                        else:
                            with st.spinner("Lade CSV-Datei..."):
                                df = load_csv(source_path=None, uploaded_file=up_file)
                                st.session_state["df"] = df
                            st.success(f"{len(df)} Zeilen wurden hochgeladen. Wählen sie im 2. Schritt den Datensatz aus.")
                    except Exception as e:
                        st.error(f"Fehler beim Laden: {e}")
                df = st.session_state.get("df", pd.DataFrame())


            with tab_manual:
                st.session_state.setdefault("manual_confirmed", False)
                st.session_state.setdefault("manual_title", "")
                st.session_state.setdefault("manual_url", "")
                st.session_state.setdefault("manual_desc", "")
           
                man_title = st.text_input("Titel", value=st.session_state.get("manual_title", ""),  placeholder="z.B. Bodenrichtwerte Sachsen-Anhalt")
                man_url = st.text_input(
                    "Dataset-URL",
                    value=st.session_state.get("manual_url", ""),
                    placeholder="https://…",
                )

                #cols_manual = st.columns([1, 1, 2], border=False)
                cols_manual = st.columns([1, 1, 2])
                with cols_manual[0]:
                    confirm_manual = st.button("Eingabe bestätigen", type="primary")
                with cols_manual[1]:
                    reset_manual = st.button("↺ Eingabe zurücksetzen", type="secondary")

                if confirm_manual:
                    if not man_title or not man_url:
                        st.warning("Bitte mindestens **Titel** und **Dataset-URL** angeben.")
                    else:
                        st.session_state["manual_title"] = man_title.strip()
                        st.session_state["manual_url"] = man_url.strip()
                        st.session_state["manual_confirmed"] = True
                        st.success("Manuelle Eingabe übernommen.")

                if reset_manual:
                    for k in ["manual_title", "manual_url", "manual_desc", "manual_confirmed"]:
                        st.session_state.pop(k, None)
                    st.info("Manuelle Eingabe zurückgesetzt.")


    # 2 Auswahl
    with col_right:
        with st.container():
            st.markdown("<h2 style='color: #1e5fcfff;'>2. Datensatz auswählen</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p>Die Auswahl wird automatisch aus Ihrer bestätigten Eingabe oder der geladenen CSV gesetzt.</p>",
                unsafe_allow_html=True,
            )

            selected_title = ""
            selected_url = ""
            selected_desc = ""

            if (
                st.session_state.get("manual_confirmed")
                and st.session_state.get("manual_title")
                and st.session_state.get("manual_url")
            ):
                selected_title = st.session_state["manual_title"]
                selected_url = st.session_state["manual_url"]
                selected_desc = st.session_state.get("manual_desc", "")
                st.success(f"Ausgewählter Datensatz (manuell): „{selected_title}“ – {selected_url}")


            else:
                df = st.session_state.get("df")
                if df is None or df.empty:
                    st.info("Laden Sie zunächst eine **CSV-Datei** hoch oder tätigen Sie eine **manuelle Eingabe**.")
                else:
                    titles = df["title"].astype(str).tolist()
                    idx = st.selectbox(
                        "Datensatz aus CSV wählen",
                        options=list(range(len(titles))),
                        format_func=lambda i: titles[i] if titles else "",
                    )
                    if titles:
                        row = df.iloc[idx]
                        selected_title = str(row.get("title", "")).strip()
                        selected_url = str(row.get("dataset", "")).strip()
                        selected_desc = str(row.get("description", ""))
                    st.caption(f"Ausgewählter Titel: {selected_title or '—'}")

    #3 Konfig
    with st.expander("3 . KI-Konfiguration (Optional)"):
        with st.container():
            #col_left2, col_mid2, col_right2 = st.columns([3.5, 2.5, 4], gap="small", border=True)
            col_left2, col_mid2, col_right2 = st.columns([3.5, 2.5, 4], gap="small")
            with col_left2:
                st.markdown("<h4>Nachricht</h4>", unsafe_allow_html=True)
                prompt_template = st.text_area(
                    "Die folgende Nachricht wird an die KI übegeben:",
                    value=prompt3_description,
                    height=400,
                    help="Rollen-/Stilvorgaben für die Beschreibung (SystemMessage).",
                )
                num_shots = st.number_input(
                    "Anzahl ähnlicher Beispiele, an denen sich die KI orientieren kann",
                    value=5,
                    min_value=0,
                    max_value=10,
                )

            with col_mid2:
                st.markdown("<h4>Parameter</h4>", unsafe_allow_html=True)
                temperature = st.slider(
                    "Temperature",
                    0.0, 2.0, 0.4, 0.1,
                    help="Steuert die Kreativität der KI: niedriger Wert = präzise, hoher Wert = variabler. Empfehlung: 0,2-0,6",
                )

                max_tokens = st.slider(
                    "Max Tokens",
                    50, 2000, 
                    1050, # Standard
                    50,
                    help="Bestimmt die Länge der Texte durch die Anzahl der Tokens: niedriger Wert = weniger Wörter; höherer Wert = mehr Wörter. Empfehlung: 300-1200.",
                )
                frequency_penalty = st.slider(
                    "Frequency Penalty",
                    -2.0, 2.0,
                    0.3, # Standard
                    0.1,
                    help="Verringert Wiederholungen: niedriger Wert = natürlicher Klang; höherer Wert = weniger Dopplungen, ggf. steifer. Empfehlung: -0.5-0,8.",
                )
                presence_penalty = st.slider(
                    "Presence Penalty",
                    -2.0, 2.0,
                    0.3, # Standard
                    0.1,
                    help="Fördert neue Begriffe/Themen: niedriger Wert = näher am Datensatz; höherer Wert = mehr Abwechslung, mögliches Abschweifen. Empfehlung: -0.5-0,5.",
                )

            with col_right2:
                st.markdown("<h4>SPARQL-Abfrage</h4>", unsafe_allow_html=True)
                edited_template = st.text_area(
                    " ",
                    value=st.session_state.get("sparql_template_text", ""),
                    height=450,
                    key="sparql_template_editor",
                    label_visibility="collapsed",
                )
                st.button("SPARQL-Query speichern", type="primary")
                st.session_state["sparql_template_text"] = edited_template

    #col_left2, col_right2 = st.columns([3, 3], gap="small", border=True)
    col_left2, col_right2 = st.columns([3, 3], gap="small")
    # 4 Beschreibung
    with col_left2:
        st.markdown("<h2 style='color: #1e5fcfff;'>4. Beschreibung erstellen</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p>Erzeugen Sie eine präzise Kurzbeschreibung, die den Inhalt des Datensatzes verständlich darstellt.</p>",
            unsafe_allow_html=True,
        )

        disabled = not bool(selected_title)
        params = ParameterPr(
            prompt_type = "Custom",
            temperature = float(temperature),
            frequency_penalty = float(frequency_penalty),
            presence_penalty = float(presence_penalty),
            max_tokens = int(max_tokens),
            num_shots = int(num_shots),
        )


        run_btn = st.button("Beschreibung generieren", type="primary", disabled=disabled)
        if st.session_state.pop("auto_run_desc", False):
            run_btn = True

        needs_warn, warn_message = parameter_warning(params)

        if run_btn:
            if needs_warn and not st.session_state.get("proceed", False):
                st.session_state["param_dialog_message"] = warn_message
                st.session_state["param_dialog_action"] = "desc"
                st.session_state["param_dialog_open"] = True
                parameterModal()
                st.stop() 
            else:
                    st.session_state["proceed"] = False

        result_text: Optional[str] = None
        used_k: Optional[int] = None
        rendered_query: Optional[str] = None
        fewshot_titles: list[dict] = []
        prompt_preview: str = ""
        fewshots: list[dict] = []


        if run_btn and selected_title:
            t0 = time.perf_counter()
            with st.spinner("Generiert eine Metadatenbeschreibung"):
                try:
                    fewshots, used_k, rendered_query = get_fewshots(
                        dataset_url=selected_url,
                        thresholds=sparql_k_thresholds,
                        limit=params.num_shots,
                        template_text_override=st.session_state.get("sparql_template_text", None),
                    )

                    fewshot_titles = [ex.get("title", "").strip() for ex in fewshots if ex.get("title")]

                    messages, prompt_preview = build_fewshot_messages(
                        system_prompt_text=prompt_template,
                        fewshots=fewshots,
                        dataset_title=selected_title,
                    )

                    generated = runLLM_messages(
                        messages,
                        temperature=params.temperature,
                        frequency_penalty=params.frequency_penalty,
                        presence_penalty=params.presence_penalty,
                        max_tokens=params.max_tokens,
                        model_name="gpt-4o-mini",
                    )
                    result_text = generated.strip()

                    st.session_state["last_generated_description"] = result_text
                    st.session_state["last_fewshot_titles"] = fewshot_titles
                    st.session_state["last_title_description"] = selected_title

                    
                except Exception as e:
                    st.error(f"Fehler: {e}")
                    result_text = None
                    prompt_preview = ""
                    fewshots = []
                    used_k = None
                    rendered_query = None

        desc_to_show = st.session_state.get("last_generated_description") or ""
        if desc_to_show:
            title_for_desc = st.session_state.get("last_title_description") or selected_title
            st.markdown(
                f"Generierte Beschreibung für<br>**{title_for_desc}**",
                unsafe_allow_html=True)
            
            st.code(desc_to_show, language="text", wrap_lines=True)

            fewshot_titles_state = st.session_state.get("last_fewshot_titles") or []
            if fewshot_titles_state:
                st.caption(
                    "Die Beschreibung wurde auf Grundlage der Beschreibungen der Titel: "
                    + ", ".join(fewshot_titles_state)
                )
            else:
                st.caption("Die Beschreibung wurde ohne ähnliche Beispiele erzeugt.")

            _desc_dl = st.download_button(
                "↘ Download Beschreibung (.txt)",
                data=desc_to_show.encode("utf-8"),
                file_name=f"{title_for_desc}_beschreibung.txt",
                mime="text/plain",
                key="download_description_txt",
            )

            meta = st.session_state.get("last_run") or {}

    # 5 Keywords
    with col_right2:
        st.markdown("<h2 style='color: #1e5fcfff;'>5. Schlagwörter erstellen</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p>Erzeugen Sie 10 aussagekräftige Schlagwörter, die den Datensatz beschreiben.</p>",
            unsafe_allow_html=True,
        )

        kw_button_disabled = not bool(selected_title)
        kw_btn = st.button("Schlagwörter generieren", type="primary", disabled=kw_button_disabled)

        if kw_btn and selected_title:
            with st.spinner("Generiert 10 Schlagwörter"):
                try:
                    keywords = generate_keywordsLLM(
                        title=selected_title,
                        model_name="gpt-4o-mini",
                        temperature=0.0,
                        max_tokens=200,
                    )
                    st.session_state["last_generated_keywords"] = keywords
                    st.session_state["last_title_keywords"] = selected_title
                except Exception as e:
                    st.error(f"Fehler bei der Keyword-Generierung: {e}")

        kws = st.session_state.get("last_generated_keywords") or []
        if kws:
            title_for_kws = st.session_state.get("last_title_keywords") or selected_title
            st.markdown(
                f"Generierte Schlagwörter für<br>**{title_for_kws}**",
                unsafe_allow_html=True)
            st.code("\n".join(kws), language="text", wrap_lines=True)

            _kw_dl = st.download_button(
                "↘ Download Schlagwörter (.txt)",
                data=("\n".join(kws)).encode("utf-8"),
                file_name= f"{title_for_kws}_keywords.txt",
                mime="text/plain",
                key="download_keywords_txt",
            )

    # 6 RDF
    with st.container(border=True):
        col_c, col_d = st.columns([4, 3])
        with col_c:
            st.markdown("<h2>6. Exportieren der generierten Metadaten</h2>", unsafe_allow_html=True)
            st.write("Exportieren Sie die erzeugte Beschreibung und die Keywords als eine DCAT-konforme RDF-Datei")
        with col_d:
            export_clicked = st.button("RDF erzeugen", type="primary")

        if export_clicked:
            if not selected_title or not (selected_url or (locals().get("man_url") or "")):
                st.warning("Bitte Titel und Dataset-URL angeben/auswählen.")
            else:
                final_desc = st.session_state.get("last_generated_description") or selected_desc or ""
                keywords = st.session_state.get("last_generated_keywords") or []

                rdf = buildRDF(
                    dataset_uri=selected_url or (locals().get("man_url") or ""),
                    title=selected_title,
                    description=final_desc,
                    keywords=keywords,
                    lang="de",
                )
                st.session_state["last_generated_rdf"] = rdf

        rdf_done = st.session_state.get("last_generated_rdf") or ""
        if rdf_done:
            st.markdown("RDF (DCAT)")
            st.code(rdf_done, language="xml", wrap_lines=True)


            rdf_downloaded = st.download_button(
                "↘ Download RDF (.rdf)",
                data=rdf_done.encode("utf-8"),
                file_name=f"{title_for_desc}_metadata.rdf",
                mime="application/rdf+xml",
                key="download_rdf_btn",
            )

            if rdf_downloaded:
                for k in [
                    "last_generated_description",
                    "last_generated_keywords",
                    "last_fewshot_titles",
                    "last_generated_rdf",
                    "last_title_description",
                    "last_title_keywords",
                ]:
                    st.session_state.pop(k, None)
                st.success("Datei wurde exportiert. Zwischenergebnisse wurden zurückgesetzt.")


if __name__ == "__main__":
    main()