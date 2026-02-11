import requests
import time
from pyproj import Transformer
import json

# ================= Konfigurasjon =================
NVDB_BASE_URL = "https://nvdbapiles.atlas.vegvesen.no"
NVDB_POSISJON_URL = f"{NVDB_BASE_URL}/vegnett/api/v4/posisjon"
NVDB_OBJEKT_URL = f"{NVDB_BASE_URL}/vegobjekter/api/v4/vegobjekter/105"

NVDB_HEADERS = {
    "X-Client": "NTNU_EiT_StudentProject_Final_v4",
    "Accept": "application/vnd.vegvesen.nvdb-v4+json"
}

transformer = Transformer.from_crs("EPSG:4326", "EPSG:5973", always_xy=True)

def log_debug(step, url, params, resp):
    print(f"\n--- DEBUG: {step} ---")
    print(f"URL: {url}")
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(f"Body: {json.dumps(data, indent=2, ensure_ascii=False)[:800]}...")
    except:
        print(f"Body: {resp.text[:200]}")

# ================= Funksjoner =================

def get_robust_speed_limit(lat, lon):
    ost, nord = transformer.transform(lon, lat)
    
    # --- STEG 1: FINN FLERE VEIKANDIDATER ---
    pos_params = {
        "nord": nord, 
        "ost": ost,
        "srid": 5973, 
        "maks_avstand": 40, # 40 meter er nok til å fange opp parallelle veier
        "maks_antall": 5,    # Hent de 5 nærmeste veiene
        "trafikantgruppe": "K"
    }
    
    try:
        pos_resp = requests.get(NVDB_POSISJON_URL, params=pos_params, headers=NVDB_HEADERS, timeout=10)
        pos_data = pos_resp.json()
        
        if not pos_data:
            return "Ingen vei funnet."

        print(f"\nSøker gjennom {len(pos_data)} mulige veier i nærheten...")

        # --- STEG 2: ITERER GJENNOM KANDIDATER ---
        for i, match in enumerate(pos_data):
            vsys = match.get('vegsystemreferanse', {})
            kortform = vsys.get('kortform', 'Ukjent')
            vls = match.get('veglenkesekvens', {})
            vls_id = vls.get('veglenkesekvensid')
            rel_pos = vls.get('relativPosisjon')
            distanse = match.get('avstand', 0)

            # Sjekk om denne spesifikke veien har en fartsgrense
            obj_params = {
                "veglenkesekvens": f"{rel_pos}@{vls_id}",
                "inkluder": "egenskaper",
                "srid": 5973
            }
            
            obj_resp = requests.get(NVDB_OBJEKT_URL, params=obj_params, headers=NVDB_HEADERS, timeout=10)
            
            if obj_resp.status_code == 200:
                obj_objekter = obj_resp.json().get("objekter", [])
                
                if obj_objekter:
                    # Vi fant en vei med registrert fartsgrense!
                    fart = None
                    for e in obj_objekter[0].get("egenskaper", []):
                        if e["id"] == 2021:
                            fart = e["verdi"]
                            break
                    
                    if fart:
                        print(f"  [Match #{i+1}] {kortform} ({int(distanse)}m unna) -> Fart funnet: {fart}")
                        return f"{fart} km/t ({kortform})"
                else:
                    print(f"  [Match #{i+1}] {kortform} ({int(distanse)}m unna) -> Ingen fartsgrense-objekt her.")

        return "Fant veier, men ingen hadde skiltet fartsgrense (mulig 50-sone/generell)."

    except Exception as e:
        return f"Feil: {str(e)}"
# ================= Kjøring =================

def main():
    points = {
        # "Tømmerdalsveien": (63.435512, 10.275317),
        # "E6": (63.326244,10.334259),
        "E6 (80)": (63.333542, 10.356348),
        "Drammensveien (Asker)": (59.8336276,10.4215051),
        "Kirkeveien (Asker)": (59.8341936,10.4241011), # Denne funker ikke
        "E18 (Asker, 60)": (59.8336673,10.4411366), # Denne strekningen på E18 har fartsgrense på 60 km/t, men det koden returnerer 90 km /t. Som det var tidligere. Strekningen er kort så den er kanskje ikke tatt med?
        "Gamle drammensvei (40)": (59.833322, 10.410803)
    }

    for name, (lat, lon) in points.items():
        print(f"\n{'#'*60}")
        print(f" ANALYSERER: {name}")
        print(f"{'#'*60}")
        resultat = get_robust_speed_limit(lat, lon)
        print(f"\n🏁 RESULTAT FOR {name}: {resultat}")
        time.sleep(1)

if __name__ == "__main__":
    main()