# Deep Reinforcement Learning für Pac-Man mit DDQN und Priorisiertem Experience Replay

Dieses Projekt implementiert einen Deep Reinforcement Learning (DRL) Agenten, der Pac-Man mit einem Double Deep Q-Network (DDQN) Algorithmus und einem priorisierten Experience Replay Buffer lernt. Der Agent wird mit TensorFlow und Keras trainiert, und die Spielumgebung wird mit Pygame erstellt.

## Inhaltsverzeichnis
- [Einführung](#einführung)
- [Funktionen](#funktionen)
- [Abhängigkeiten](#abhängigkeiten)
- [Installation](#installation)
- [Nutzung](#nutzung)
- [Spielumgebung](#spielumgebung)
- [Reinforcement Learning Agent](#reinforcement-learning-agent)
- [Priorisierter Replay Buffer](#priorisierter-replay-buffer)
- [Training](#training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Ergebnisse](#ergebnisse)
- [Zukünftige Verbesserungen](#zukünftige-verbesserungen)
- [Lizenz](#lizenz)

## Einführung
Dieses Projekt zielt darauf ab, einen KI-Agenten zu erstellen, der Pac-Man durch Lernen optimaler Aktionen durch Versuch und Irrtum spielen kann. Der Agent verwendet den Double Deep Q-Network (DDQN) Algorithmus, der eine Verbesserung gegenüber dem traditionellen Deep Q-Network (DQN) darstellt. Der DDQN hilft, die Überschätzung von Q-Werten zu reduzieren, was zu einem stabileren Lernen führt. Zusätzlich nutzt die Implementierung einen priorisierten Experience Replay Buffer, der es dem Agenten ermöglicht, effektiver aus Erfahrungen mit hohen Fehlern zu lernen.

## Funktionen
- Implementiert eine Pac-Man-Spielumgebung mit Pygame.
- Verwendet einen DDQN-Agenten, der mit TensorFlow und Keras implementiert ist.
- Implementiert einen priorisierten Experience Replay Buffer für effizienteres Lernen.
- Enthält Hyperparameter-Tuning für optimierte Leistung.
- Visualisiert das Spiel während des Trainings zur Beobachtung und Fehlerbehebung.
- Ermöglicht das Speichern und Laden trainierter Modelle.

## Abhängigkeiten
- Python 3.6+
- Pygame
- NumPy
- TensorFlow 2.x
- Keras

Installieren Sie die erforderlichen Pakete:

```bash
pip install pygame numpy tensorflow
```

## Installation
1. Klonen Sie das Repository:
```bash
git clone https://github.com/kruemmel-python/PacMaster_AI.git
cd PacMaster_AI
```
2. Installieren Sie die Abhängigkeiten (siehe Abschnitt "Abhängigkeiten" oben).

## Nutzung
Um den Agenten zu trainieren und das Pac-Man-Spiel zu spielen, führen Sie das Hauptskript aus:

```bash
python main.py
```

Der Trainingsprozess gibt Episode-Informationen aus, und das Spiel-Fenster visualisiert das aktuelle Gameplay während des Trainings.

## Spielumgebung
Die Spielumgebung wird mit Pygame implementiert. Wichtige Merkmale sind:
- Dynamische Positionierung von Pac-Man, Geistern, Pellets und Power-Pellets
- Geisterbewegung basierend auf Pac-Mans Position.
- Kollisionserkennung zwischen Pac-Man und Spielobjekten.
- Power-Up-Modus, der durch den Verzehr von Power-Pellets ausgelöst wird.
- Belohnungssystem basierend auf Spielaktionen.

Der Zustand des Spiels wird als 17-dimensionaler Vektor dargestellt, der Folgendes umfasst:
- Normalisierte Pac-Man-Position
- Normalisierte Geisterpositionen
- Relative Positionen des nächsten Pellets und Power-Pellets
- Normalisierte Distanzen zu jedem Geist (max. 4)
- Ein Indikator, ob Pac-Man im Power-Up-Modus ist.

## Reinforcement Learning Agent
Der Agent wird mit TensorFlow und Keras erstellt. Er verfügt über:
- Eine Double Deep Q-Network (DDQN) Architektur zur Approximation der Q-Funktion.
- Die Netzwerkarchitektur umfasst gemeinsame dichte Schichten mit Layer Normalization sowie separate Value- und Advantage-Streams.
- Der Adam-Optimizer wird mit einer festgelegten Lernrate verwendet.
- Epsilon-Greedy-Policy für den Trade-off zwischen Exploration und Exploitation.
- Zielnetzwerk zur Stabilisierung des Trainings.

## Priorisierter Replay Buffer
Der Experience Replay Buffer verwendet eine priorisierte Methode, die dem Agenten hilft, effektiver zu lernen:
- Erfahrungen mit höheren Temporal-Difference (TD) Fehlern werden häufiger gesampelt.
- Die Prioritäten werden aktualisiert, nachdem der Agent aus einem Batch von Erfahrungen gelernt hat.
- Die Sampling-Wahrscheinlichkeit wird durch die Prioritäten jeder Erfahrung definiert.
- Der Buffer enthält einen Mechanismus zum Annealing der Importance Sampling Weights (Beta) über die Zeit.

## Training
Der Agent wird über eine definierte Anzahl von Episoden trainiert, während denen:
- Der Agent mit der Umgebung interagiert, Aktionen ausführt und Belohnungen erhält.
- Die Erfahrungen werden in einem priorisierten Replay Buffer gespeichert.
- Der Agent lernt aus gesampelten Erfahrungen und aktualisiert die Gewichte seines Netzwerks.
- Die Gewichte des Zielnetzwerks werden in festgelegten Intervallen aktualisiert.
- Das Modell des Agenten wird nach jeder Episode gespeichert.

## Hyperparameter Tuning
Das Skript enthält Hyperparameter-Tuning für den priorisierten Replay Buffer. Es testet verschiedene Kombinationen der `alpha` und `beta_start` Parameter:
- `alpha` steuert die Priorisierung des Replay Buffers.
- `beta_start` steuert das anfängliche Importance Sampling Weight.

Die besten Hyperparameter werden nach dem Tuning-Prozess ausgegeben.

## Ergebnisse
Der Trainingsprozess gibt die durchschnittliche Belohnung pro Episode und die dafür benötigte Dauer aus.

## Zukünftige Verbesserungen
- Implementieren Sie eine robustere Zustandsdarstellung der Spielumgebung.
- Testen Sie verschiedene neuronale Netzwerkarchitekturen und Schichtkombinationen.
- Fügen Sie eine Visualisierung der Daten des priorisierten Replay Buffers während des Trainings hinzu.
- Verbessern Sie die Geschwindigkeit der Pygame-Umgebung für schnelleres Training.

## Lizenz
Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Details finden Sie in der `LICENSE`-Datei.
