const categories = [
  "Usługi reklamowe",
  "Pasmanteria",
  "Lodziarnia",
  "Dostawca alarmów samochodowych",
  "Sklep",
  "Mechanik samochodowy",
  "Warsztat samochodowy",
  "Usługa transportowa",
  "Firma transportowa",
  "Usługi informatyczne",
  "Serwis komputerowy",
  "Naprawa i serwis komputerów",
  "Serwis komputerów",
  "Sklep komputerowy",
  "Komputery",
  "Pomoc drogowa",
  "Holowanie",
  "Dostawca sprzętu do pozyskiwania energii słonecznej",
  "Dostawca energii odnawialnej",
  "Instalacja klimatyzacji",
  "Wulkanizacja",
  "Sklep z oponami i felgami",
  "Prawnik",
  "Usługi prawne",
  "Adwokat",
  "Kancelaria",
  "Prawnik – ubezpieczenia",
  "Prawnik – obrażenia osób",
  "Pośrednik kredytów hipotecznych",
  "Pośrednik finansowy",
  "Doradca finansowy",
  "Salon piękności",
  "Salon manicure i pedicure",
  "Producent",
  "Obróbka metali",
  "Siedziba firmy",
  "Usługi dla biznesu",
  "Artykuły metalowe",
  "Salon kosmetyczny",
  "Sklep samochodowy",
  "Oleje i smary",
  "Optyk",
  "Hydraulik",
  "Warsztat tuningu samochodów",
  "Ślusarz",
  "Dorabianie kluczy",
  "Wynajem samochodu z kierowcą",
  "Pracownia rentgenowska",
  "Klinika stomatologiczna",
  "Taksówki",
  "Stacja",
  "Stomatolog",
  "Serwis samochodów ciężarowych",
  "Restauracja",
  "Fast food",
  "Kuchnia włoska",
  "Bistro",
  "Pizzeria",
  "Restauracja dla rodzin",
  "Obiekt przeznaczony do organizacji imprez",
  "Kuchnia polska",
  "Wyposażenie przemysłowe",
  "Sklep firmowy",
  "Sklep z oprogramowaniem komputerowym",
  "Dostawca wyposażenia sklepów",
  "Przyciemnianie szyb",
  "Agencja reklamowa",
  "Szyldy reklamowe",
  "Drukarnia",
  "Reklama świetlna",
  "Studio tatuażu",
  "Projektowanie stron WWW",
  "Agencja marketingowa",
  "Agencja interaktywna",
  "Hosting internetowy",
  "Punkt ksero",
  "Weterynarz",
  "Lekarz",
  "Sklep z używanymi oponami",
  "Sklep z akumulatorami",
  "Prawnik – prawo rodzinne",
  "Przeprowadzki",
  "Sklep rowerowy",
  "Naprawa rowerów",
  "Myjnia samochodowa",
  "Suknie ślubne",
  "Sklep odzieżowy",
  "Doradca marketingowy",
  "Wykonawca instalacji grzewczych, klimatyzacyjnych i wentylacyjnych",
  "Usługi ciepłownicze",
  "Firma budowlana",
  "Wykonawca izolacji",
  "Sushi",
  "Kuchnia wietnamska",
  "Restauracja europejska (kuchnia nowoczesna)",
  "Salon fryzjerski",
  "Fryzjer",
  "Komis samochodowy",
  "Salon samochodowy",
  "Depilacja laserowa",
  "Przychodnia lekarska",
  "Kosmetyczka",
  "Centrum medyczne",
  "Klinika specjalistyczna",
  "Chirurg plastyczny",
  "Spa i pielęgnacja twarzy",
  "Serwis klimatyzacji",
  "Wyposażenie łazienek",
  "Hotel",
  "Zakwaterowanie z wyżywieniem we własnym zakresie",
  "Apartamenty wczasowe",
  "SPA",
  "Pensjonat",
  "Doradztwo kredytowe",
  "Neurolog",
  "Dostawca kruszyw",
  "Dostawca materiałów budowlanych",
  "Skład materiałów budowlanych",
  "Szkoła języka angielskiego",
  "Szkoła językowa",
  "Szkoła",
  "Prawnik – prawo pracy",
  "Agencja nieruchomości",
  "Czyszczenie samochodów",
  "Renowacja pojazdów",
  "Czyszczenie tapicerki",
  "Sklep z witaminami i suplementami",
  "Sklep sportowy",
  "Gabinet lekarski",
  "Dostawca systemów ochrony",
  "Montaż instalacji LPG",
  "Blacharstwo samochodowe",
  "Szyby samochodowe",
  "Dostawca okien",
  "Sklep internetowy",
  "Centrum szkoleniowe",
  "Centrum biznesowe",
  "Notariusz",
  "Adwokat sądowy",
  "Biuro rachunkowe",
  "Fundacja",
  "Agencja ubezpieczeniowa",
  "Prawnik – ochrona majątku",
  "Prawnik medyczny",
  "Sprzedaż internetowa",
  "Fotograf weselny",
  "Fotograf",
  "Zakład fotograficzny",
  "Przeprowadzki i usługi magazynowe",
  "Automatyka",
  "Montaż okien",
  "Dostawca drzwi garażowych",
  "Elektryk",
  "Wykonawca ogrodzeń",
  "Wykonawca",
  "Wypożyczalnia samochodów",
  "Okulista",
  "Hurtownia kosmetyków",
  "Drogeria",
  "Sklep z kosmetykami",
  "Sklep z meblami kuchennymi",
  "Noclegi",
  "Noclegi pod dachem",
  "Noclegi ze śniadaniem",
  "Agroturystyka",
  "Protetyk stomatologiczny",
  "Protetyka",
  "Pracownia dentystyczna",
  "Kawiarnia",
  "Producent oprogramowania",
  "Naprawa telefonów komórkowych",
  "Sklep RTV",
  "Adwokat rozwodowy",
  "Materiały podłogowe",
  "Materiały drewniane",
  "Wykonywanie podłóg",
  "Dostawca drzwi",
  "Sklep dla majsterkowiczów",
  "Sklep z żaluzjami",
  "Sklep z oknami plastikowymi",
  "Budowa domów",
  "Sklep meblowy",
  "Prawnik – prawo upadłościowe",
  "Ochrona przed szkodnikami",
  "Usługi sprzątania",
  "Czyszczenie kominów",
  "Usługi fotograficzne",
  "Dostawca usług internetowych",
  "Sklep turystyczny",
  "Skup złomu i surowców wtórnych",
  "Stowarzyszenie lub organizacja",
  "Wyroby hutnicze",
  "Konstrukcje stalowe",
  "Czyszczenie dywanów",
  "Pralnia",
  "Firmy sprzątające",
  "Remonty",
  "Usługi remontowo-budowlane",
  "Sklep z tapetami",
  "Producent mebli",
  "Fizjoterapeuta",
  "Trener osobisty",
  "Dietetyk",
  "Specjalista rehabilitacji",
  "Klub fitness",
  "Szkoła nauki jazdy",
  "Wideofilmowanie",
  "Księgarnia",
  "Sklep fotograficzny",
  "Stacja paliw",
  "Didżej",
  "Organizator imprez",
  "Doradca podatkowy",
  "Transfer lotniskowy",
  "Księgowy",
  "Usługi księgowe",
  "Biuro podatkowe",
  "Sklep z telefonami komórkowymi",
  "Sklep z zabawkami",
  "Biuro nieruchomości",
  "Wywóz odpadów komunalnych",
  "Zarządzanie odpadami",
  "Willa",
  "Sklep z konopiami",
  "Sklep zielarski",
  "Pośrednik w obrocie nieruchomościami",
  "Laboratorium",
  "Wynajem domków letniskowych",
  "Projektant wnętrz",
  "Ramen",
  "Prywatny detektyw",
  "Sklep z produktami dla dzieci",
  "Złomowisko",
  "Złomowanie samochodów",
  "Obrońca w sprawach karnych",
  "Meble biurowe",
  "Psycholog",
  "Psychoterapeuta",
  "Punkt poboru opłat",
  "Masażysta",
  "Fryzjer dla zwierząt",
  "Mycie ciśnieniowe",
  "Pizza na wynos",
  "Dostawca balonów",
  "Program leczenia alkoholizmu",
  "Klub sportowy",
  "Serwis sprzętu rolniczego",
  "Świece i znicze",
  "Logopeda",
  "Przedsiębiorstwo wodociągowe",
  "Laboratorium medyczne",
  "Kantor",
  "Osiedle mieszkaniowe",
  "Deweloper",
  "Budownictwo mieszkalne",
  "Brukarstwo",
  "Naprawa telefonów",
  "Sklep z alarmami",
  "Strzelnica",
  "Naprawa wyrobów skórzanych",
  "Sklep z odzieżą roboczą",
  "Sklep z materiałami dekarskimi",
  "Dostawca farb",
  "Sklep z ogrodzeniami",
  "Sklep z upominkami",
  "Dekarz",
  "Stadion",
  "Sklep zoologiczny",
  "Sklep z materacami",
  "Centrum paintballowe",
  "Sklep obuwniczy",
  "Konsultant ds. nieruchomości",
].map((category, index) => ({
  label: category,
  id: index
}));

export default categories;
