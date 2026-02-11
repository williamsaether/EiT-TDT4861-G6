from speed_limit.nvdb_speed import get_speed_limit_data

def run_tests():
    test_cases = [
        {"navn": "E6 Sluppen", "lat": 63.333542, "lon": 10.356348, "forventet_fart": 80},
        {"navn": "Asker Kirkeveien", "lat": 59.834185, "lon": 10.428984, "forventet_fart": 50},
        {"navn": "E18 Asker", "lat": 59.8336673, "lon": 10.4411366, "forventet_fart": 90},
        {"navn": "Gamle Drammensvei", "lat": 59.833322, "lon": 10.410803, "forventet_fart": 40},
        {"navn": "Tømmerdalsveien", "lat": 63.435512, "lon": 10.275317, "forventet_fart": 50},
        {"navn": "E6", "lat": 63.326244, "lon": 10.334259, "forventet_fart": 100},
    ]

    print(f"{'TESTNAVN':<20} | {'STATUS':<10} | {'FUNNET':<15} | {'FORVENTET'}")
    print("-" * 65)

    for case in test_cases:
        res = get_speed_limit_data(case["lat"], case["lon"])
        
        if res and res["fartsgrense"] == case["forventet_fart"]:
            status = "✅ OK"
        else:
            status = "❌ FEIL"
        
        fart_funnet = res["fartsgrense"] if res else "Ingen"
        print(f"{case['navn']:<20} | {status:<10} | {fart_funnet:<15} | {case['forventet_fart']}")

if __name__ == "__main__":
    run_tests()