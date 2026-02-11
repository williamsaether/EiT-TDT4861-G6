import time
import nvdb_speed

def simulate_drive():
    # Nullstill global variabel før turen starter
    nvdb_speed.LAST_VEGLENKE_ID = None 

    # En liste med koordinater som simulerer en kjøretur (eksempel: fra en vei til en annen)
    # Her kan du legge inn punkter fra Google Maps e.l.
    route = [
        (59.835764, 10.423201),
        (59.835746, 10.423010),
        (59.835727, 10.422836),
        (59.835735, 10.422685), # Kryss 1
        (59.835705, 10.422611),
        (59.835671, 10.422493), #Kryss 2
        (59.835673, 10.422376),
    ]

    current_road = None
    current_speed_limit = None  

    print("🚀 Starter kjøresimulering...\n")

    for i, (lat, lon) in enumerate(route):
        print(f"📍 Posisjon {i+1}: ({lat}, {lon})")
        
        data = nvdb_speed.get_speed_limit_data(lat, lon)
        
        if data:
            # Sjekk om vi har byttet vei eller fartsgrense
            if data['vei'] != current_road or data['fartsgrense'] != current_speed_limit:
                print(f"🔔 ENDRING OPPDAGET!")
                print(f"   🛣️  Vei: {data['vei']}")
                print(f"   🚦 Fartsgrense: {data['fartsgrense']} km/t")
                
                current_road = data['vei']
                current_speed_limit = data['fartsgrense']
            else:
                print(f"   --- Fortsetter på {data['vei']} ({data['fartsgrense']} km/t) ---")
        else:
            print("   ⚠️  Kunne ikke finne veidata for dette punktet.")

        print("-" * 40)
        time.sleep(2) # Vent 2 sekunder mellom hvert "API-kall" for å simulere kjøring

    print("\n🏁 Simulering avsluttet.")

if __name__ == "__main__":
    simulate_drive()