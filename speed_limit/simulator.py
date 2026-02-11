import time
from nvdb_speed import get_speed_limit_data

def simulate_drive():
    # En liste med koordinater som simulerer en kjøretur (eksempel: fra en vei til en annen)
    # Her kan du legge inn punkter fra Google Maps e.l.
    route = [
        (63.435512, 10.275317), # Tømmerdalsveien
        (63.435800, 10.276000), # Tømmerdalsveien videre
        (63.333542, 10.356348), # Hopper til E6 (simulert sving/flytting)
        (63.334000, 10.357000), # E6 videre
        (59.833322, 10.410803)  # Hopper til Gamle Drammensvei
    ]

    current_road = None
    current_speed_limit = None

    print("🚀 Starter kjøresimulering...\n")

    for i, (lat, lon) in enumerate(route):
        print(f"📍 Posisjon {i+1}: ({lat}, {lon})")
        
        data = get_speed_limit_data(lat, lon)
        
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