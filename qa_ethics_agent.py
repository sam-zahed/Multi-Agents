# === Dummy-Basisklasse zur Kompatibilität mit Agent-Konzept (falls kein echter LangChain-Agent verwendet wird) ===
class Agent:
    def __init__(self, name=None, instructions=None):
        self.name = name  # Anzeigename des Agenten
        self.instructions = instructions  # Interne Beschreibung bzw. Verhaltenserwartung

# === Funktion zur Überprüfung von Antworten auf Qualität, Ethik und Bias ===
def check_facts_and_ethics(answer, sources):
    # Hier könnten GPT-Modelle, OpenAI Moderation API oder eigene Heuristiken genutzt werden
    warnings = []
    if not answer or not isinstance(answer, str) or len(answer) == 0:
        warnings.append("⚠️ Keine Antwort erhalten.")
        return warnings
    if not sources or len(sources) == 0:
        warnings.append("⚠️ Keine Quellenangabe gefunden.")
    if "kann ich nicht" in answer.lower() or "unbekannt" in answer.lower():
        warnings.append("⚠️ Antwort ist unvollständig oder unsicher.")
    # Sehr einfache Bias-Erkennung: übergeneralisierende Begriffe
    if "immer" in answer.lower() or "nie" in answer.lower():
        warnings.append("⚠️ Möglicher Bias in der Formulierung erkannt.")
    # Erweiterbar: weitere Checks wie Faktenprüfung, Sentiment etc.
    return warnings

# === Agent zur Durchführung der Ethik- und QA-Prüfung von Antworten ===
class QA_EthicsAgent(Agent):
    def __init__(self):
        super().__init__(
            name="QA & Ethics Reviewer",  # Anzeigename
            instructions="Prüft Antworten auf Fakten, Quellen, Bias und Ethik."  # Beschreibung für die Aufgabe des Agenten
        )

    def run(self, answer, sources):
        warnings = check_facts_and_ethics(answer, sources)
        if warnings:
            return "\n".join(warnings)  # Warnungen gesammelt als String zurückgeben
        return "✅ Antwort besteht die QA/Ethik-Prüfung."  # Bestätigung bei sauberer Antwort

# === Instanzierung des QA/Ethik-Agenten für externe Nutzung ===
qa_ethics_agent = QA_EthicsAgent()
