import json
from pathlib import Path
import logging

INPUT_FILE_DIR = Path("../../all_json")
INPUT_FILE = Path("../../all_json/63963.json")
OUTPUT_FILE = Path("output/63963_cricket_corpus.jsonl")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Helper functions
# -------------------------------

def describe_ball(over, ball):
    batter = ball.get("batter")
    bowler = ball.get("bowler")
    runs = ball.get("runs", {}).get("batter", 0)
    total = ball.get("runs", {}).get("total", 0)
    wicket = ball.get("wickets")

    if wicket:
        player_out = wicket[0]["player_out"]
        kind = wicket[0]["kind"]
        return f"Over {over}: {bowler} dismisses {player_out} with a {kind}."

    if runs == 0:
        return f"Over {over}: {bowler} bowls a tight delivery to {batter}. No run."
    elif runs == 1:
        return f"Over {over}: {batter} works {bowler} into the gap for a single."
    elif runs == 2:
        return f"Over {over}: {batter} picks up two runs off {bowler}."
    elif runs == 3:
        return f"Over {over}: {batter} runs hard to collect three."
    elif runs == 4:
        return f"Over {over}: {batter} finds the boundary off {bowler}."
    elif runs == 6:
        return f"Over {over}: {batter} launches {bowler} for a six."
    else:
        return f"Over {over}: {batter} scores {runs} runs off {bowler}."


def summarize_over(over_number, deliveries):
    total_runs = sum(d.get("runs", {}).get("total", 0) for d in deliveries)
    wickets = sum(1 for d in deliveries if d.get("wickets"))
    bowlers = list({d.get("bowler") for d in deliveries})

    summary = f"Over {over_number} summary: "
    summary += f"{', '.join(bowlers)} concedes {total_runs} runs."

    if wickets:
        summary += f" {wickets} wicket(s) fell in the over."

    return summary


def phase_of_over(over):
    if over < 6:
        return "powerplay"
    elif over < 15:
        return "middle overs"
    else:
        return "death overs"


def analyze_phase(team, phase, balls):
    runs = sum(b.get("runs", {}).get("total", 0) for b in balls)
    overs = max(1, len(balls) / 6)

    run_rate = round(runs / overs, 2)

    return (
        f"{team} scored at {run_rate} runs per over during the {phase}, "
        f"showing {'aggression' if run_rate > 7 else 'control'}."
    )

def match_summary(match, file_name):
    try:
        event_name = match.get("info").get("event").get("name", "Unknown event")
    except:
        logging.warning(f"Event name not found: {file_name}")
        return

    event_venue = match.get("info").get("venue", "Unknown venue")
    event_type = match.get("info").get("match_type", "Unknown type")
    event_city = match.get("info").get("city", "Unknown city")
    event_city = match.get("info").get("city", "Unknown city")
    event_season = match.get("info").get("season", "Unknown season")

    event_innings_count = len(match.get("innings"))
    event_start_date = event_end_date = match.get("info").get("dates")[0]

    if event_innings_count > 1:
        try:
            event_end_date = match.get("info").get("dates")[event_innings_count-1]
        except:
            event_end_date = match.get("info").get("dates")[len(match.get("info").get("dates"))-1]

    toss_winner = match.get("info").get("toss").get("winner")
    toss_decision = match.get("info").get("toss").get("decision")

    if toss_decision == "bat":
        toss_decision = "batting"
    else:
        toss_decision = "bowling"
    
    player_of_match = match.get("info").get("player_of_match", [""])
    event_result = match.get("info").get("outcome")

    if "result" in event_result:
        event_outcome = event_result.get("result")
    else:
        try:
            by = event_result.get("by")
            if "runs" in by:
                event_outcome = event_result.get("winner") + " woned by " + str(event_result.get("by").get("runs")) + " runs"
            elif "wickets" in by:
                event_outcome = event_result.get("winner") + " woned by " + str(event_result.get("by").get("wickets", 0)) + " wickets"
        except:
            event_outcome = event_result.get("winner") + " woned and method was awarded"

    officials_umpires = ",".join(list(match.get("info").get("officials",{"umpires": ""}).get("umpires", "")))
    tv_umpires = ",".join(list(match.get("info").get("officials",{"tv_umpires": ""}).get("tv_umpires", "")))
    match_refrees = ",".join(list(match.get("info").get("officials",{"match_referees": ""}).get("match_referees", "")))
    team_type = match.get("info").get("team_type", "Unknown team_type")
    teams = match.get("info").get("teams")

    players = match.get("info").get("players")
    match_players = " "

    for t_name, player in players.items():
        match_players = match_players + " " + f"Player for {t_name}: " + ", ".join(player) + "."

    final_str = f"""A {event_type} series between {teams[0]} vs {teams[1]} started on {event_start_date} and ended on {event_end_date}. The series name is '{event_name}' and venue is '{event_venue}'. In this match {toss_winner} won the toss and choose to {toss_decision}. In this {event_season} season, Man of the match was '{player_of_match[0]}' and the result was, {event_outcome}. The officials umpires were {officials_umpires}, tv umpires were {tv_umpires} and the match refree was {match_refrees}. This series was {team_type} match. {match_players}."""

    return final_str

# -------------------------------
# Main conversion logic
# -------------------------------

def convert_match_to_text(match):
    texts = []

    for inning in match.get("innings", []):
        team = inning.get("team", "Unknown Team")

        phase_balls = {
            "powerplay": [],
            "middle overs": [],
            "death overs": []
        }

        for over_data in inning.get("overs", []):
            over_number = over_data.get("over")
            deliveries = over_data.get("deliveries", [])

            # Ball-level commentary
            for ball in deliveries:
                texts.append(describe_ball(over_number, ball))

                phase = phase_of_over(over_number)
                phase_balls[phase].append(ball)

            # Over summary
            texts.append(summarize_over(over_number, deliveries))

        # Phase analysis
        for phase, balls in phase_balls.items():
            if balls:
                texts.append(analyze_phase(team, phase, balls))

        # Innings summary
        total_runs = sum(
            b.get("runs", {}).get("total", 0)
            for phase in phase_balls.values()
            for b in phase
        )

        texts.append(
            f"{team} finished the innings with a total of {total_runs} runs."
        )

    return texts


# -------------------------------
# main
# -------------------------------

if __name__ == "__main__":

    dir_path = Path(INPUT_FILE_DIR)
    files = list(dir_path.glob("*.json"))

    for i_file in files:
        
        with open(i_file, "r", encoding="utf-8") as f:
            match_data = json.load(f)
        print(f"------file name------ {i_file} \n")
        match_summary_text = match_summary(match_data, i_file)
        print("------------------------\n")
        
        cricket_texts = convert_match_to_text(match_data)
        cricket_texts.insert(0, match_summary_text)

        file_name = str(i_file).split("\\")[3].split(".")[0]
        print(f"file name: {file_name}")
        o_file = f"output2/{file_name}_cricket_corpus.jsonl"
        
        with open(o_file, "w", encoding="utf-8") as f:
            for text in cricket_texts:
                f.write(json.dumps({"text": text}) + "\n")

        print(f"Generated {i_file} - {len(cricket_texts)} cricket text samples")
