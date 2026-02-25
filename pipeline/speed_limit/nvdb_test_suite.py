from nvdb_speed import get_speed_limit_data

def run_tests():
    test_cases = [
        {"navn": "E6 Sluppen", "lat": 63.333542, "lon": 10.356348, "forventet_fart": 80},
        {"navn": "Asker Kirkeveien", "lat": 59.834185, "lon": 10.428984, "forventet_fart": 50},
        {"navn": "E18 Asker", "lat": 59.8336673, "lon": 10.4411366, "forventet_fart": 90},
        {"navn": "Gamle Drammensvei", "lat": 59.833322, "lon": 10.410803, "forventet_fart": 40},
        {"navn": "Tømmerdalsveien", "lat": 63.435512, "lon": 10.275317, "forventet_fart": 50},
        {"navn": "E6", "lat": 63.326244, "lon": 10.334259, "forventet_fart": 100},
        {"navn": "Gamle Drammensvei (kryss)", "lat": 59.835707, "lon": 10.422574, "forventet_fart": 40},
        {"navn": "Øvre Askerhagen", "lat": 59.835592, "lon": 10.422452, "forventet_fart": 50},
    ]

    # 1. Finn lengden på det lengste navnet (minimum 20 for å passe headeren)
    max_name_len = max(len(case["navn"]) for case in test_cases)
    col_width = max(max_name_len, 8) + 2  # +2 for litt luft før streken

    # 2. Lag header med dynamisk bredde
    header = f"{'TESTNAVN':<{col_width}} | {'STATUS':<11} | {'FUNNET':<15} | {'FORVENTET'}"
    print(header)
    print("-" * len(header))

    for case in test_cases:
        res = get_speed_limit_data(case["lat"], case["lon"])
        
        # Sjekk resultat (håndterer både dict og None)
        fart_funnet = res.get("fartsgrense") if (res and isinstance(res, dict)) else None
        
        if fart_funnet == case["forventet_fart"]:
            status = "✅ OK"
        else:
            status = "❌ FEIL"
        
        display_fart = fart_funnet if fart_funnet else "Ingen"
        
        # 3. Print raden med samme dynamiske bredde
        print(f"{case['navn']:<{col_width}} | {status:<10} | {display_fart:<15} | {case['forventet_fart']}")

if __name__ == "__main__":
    run_tests()