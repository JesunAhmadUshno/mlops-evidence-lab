"""Level 2, Build: the synthetic corpus.

20 short documents across 7 topics. Synthetic teaching data by
design, authored for this level. Unlike the Learn corpus, topics
deliberately share vocabulary so retrieval is harder: water words
appear in the water, energy, animals, and plants documents; gravity
and moon appear in both the earth and the tides documents; bees and
pollen appear in both the plants and the animals documents; the
three Northlight library documents share the library, members, and
books vocabulary and differ only in their facts. Every document is
labelled with its topic so the overlap is inspectable.
"""

DOCUMENTS = [
    {
        "id": "water-cycle",
        "topic": "water",
        "text": (
            "The water cycle moves water between the earth, the oceans, "
            "and the sky. Heat from the sun causes evaporation from "
            "oceans, lakes, and rivers. Water vapour cools high in the "
            "sky, condensation forms clouds, and precipitation falls "
            "as rain or snow. The fallen water collects in rivers and "
            "groundwater and the cycle begins again."
        ),
    },
    {
        "id": "rivers-and-lakes",
        "topic": "water",
        "text": (
            "Rivers carry fresh water from hills and mountains down to "
            "lakes and the sea. A river begins as a small stream and "
            "grows as rain and melting snow join it. Lakes are still "
            "water held in low ground, and many lakes store drinking "
            "water for nearby towns. Rivers shape valleys by carrying "
            "stones and soil downstream over long years."
        ),
    },
    {
        "id": "ocean-tides",
        "topic": "water",
        "text": (
            "Ocean tides are the daily rise and fall of sea water along "
            "coasts. The gravity of the moon pulls on ocean water and "
            "raises a bulge that causes high tide. When the pull of the "
            "moon is weakest the water level drops and low tide leaves "
            "rock pools exposed. Most coasts have two high tides and "
            "two low tides each day."
        ),
    },
    {
        "id": "photosynthesis",
        "topic": "plants",
        "text": (
            "Photosynthesis is how green plants make food using "
            "sunlight. Plants absorb carbon dioxide from the air and "
            "water from the soil through their roots. Inside leaves, "
            "sunlight powers a chemical change that makes glucose "
            "sugar, and plants release oxygen into the air. Most "
            "photosynthesis happens in leaves, which are broad and "
            "flat to catch sunlight."
        ),
    },
    {
        "id": "pollination",
        "topic": "plants",
        "text": (
            "Pollination is how flowering plants move pollen from "
            "flower to flower so seeds can form. Bees, butterflies, "
            "and wind carry pollen as they visit flowers for nectar. "
            "When pollen lands on the stigma of a matching flower, "
            "fertilisation can begin and a seed starts to grow. Many "
            "food crops depend on insect pollination for their harvest."
        ),
    },
    {
        "id": "seed-dispersal",
        "topic": "plants",
        "text": (
            "Seed dispersal spreads plant seeds away from the parent "
            "plant so new plants have room to grow. Wind carries light "
            "seeds with wings or fine hairs across fields. Animals "
            "carry seeds that stick to fur or pass through their "
            "bodies after eating fruit. Water carries floating seeds "
            "such as coconuts along rivers and across seas to new shores."
        ),
    },
    {
        "id": "gravity",
        "topic": "earth",
        "text": (
            "Gravity is a force that pulls objects toward each other. "
            "Isaac Newton described gravity as a universal attraction "
            "between masses. The gravity of Earth pulls objects toward "
            "its centre, which is why dropped objects fall down and "
            "rain falls from clouds. The moon stays in orbit around "
            "Earth because gravity holds it, and the gravity of the "
            "moon raises ocean tides."
        ),
    },
    {
        "id": "volcanoes",
        "topic": "earth",
        "text": (
            "A volcano is an opening where molten rock can erupt from "
            "below the surface of the earth. Magma is molten rock "
            "stored underground, and magma becomes lava when it "
            "reaches the surface during an eruption. Volcanic ash can "
            "cover fields and roofs, and lava flows can change the "
            "landscape for many years. Some volcanoes erupt often, "
            "and others sleep for centuries between eruptions."
        ),
    },
    {
        "id": "earthquakes",
        "topic": "earth",
        "text": (
            "An earthquake is a sudden shaking of the ground caused "
            "by rocks moving below the surface of the earth. Stress "
            "builds where giant rock plates press against each other "
            "until the rocks slip and release energy as waves. The "
            "point where the slip starts underground is called the "
            "focus, and the place on the surface above it is the "
            "epicentre. Strong earthquakes can crack roads, damage "
            "buildings, and trigger landslides on steep hills."
        ),
    },
    {
        "id": "bees",
        "topic": "animals",
        "text": (
            "Honey bees collect nectar and pollen from flowers to "
            "feed their colony. Bees communicate the location of "
            "flowers by performing a waggle dance inside the hive. "
            "A colony has one queen bee, many worker bees, and a "
            "smaller number of drone bees. Bees carry pollen between "
            "flowers as they feed, so bees help pollination while "
            "they gather food."
        ),
    },
    {
        "id": "bird-migration",
        "topic": "animals",
        "text": (
            "Bird migration is the seasonal journey many birds make "
            "between breeding grounds and winter grounds. Birds "
            "navigate using the sun, the stars, and the magnetic "
            "field of the earth. Migration lets birds follow food "
            "supplies such as insects and seeds as seasons change. "
            "Some migrating birds travel thousands of kilometres and "
            "return to the same nesting place each year."
        ),
    },
    {
        "id": "salmon-run",
        "topic": "animals",
        "text": (
            "The salmon run is the journey adult salmon make from the "
            "ocean back to the rivers where they were born. Salmon "
            "swim upstream against the river current and leap over "
            "small waterfalls. When salmon reach their home stream "
            "they lay eggs in gravel beds and the young fish hatch "
            "there. Bears and birds gather along rivers to feed on "
            "salmon during the run."
        ),
    },
    {
        "id": "solar-power",
        "topic": "energy",
        "text": (
            "Solar power turns sunlight into electricity using solar "
            "panels. A solar panel contains photovoltaic cells that "
            "release electrons when sunlight strikes them, and the "
            "flow of electrons is electricity. Solar farms place many "
            "panels in open fields, and rooftop panels power single "
            "homes. Solar power makes no electricity at night, so "
            "batteries store daytime energy for evening use."
        ),
    },
    {
        "id": "wind-power",
        "topic": "energy",
        "text": (
            "Wind power turns moving air into electricity using wind "
            "turbines. Wind spins the blades of a turbine, and a "
            "generator inside changes the spin into electricity. Wind "
            "farms stand on hills and coasts where winds blow steadily "
            "through the year. Wind power depends on the weather, so "
            "its output rises and falls as the wind changes."
        ),
    },
    {
        "id": "hydro-power",
        "topic": "energy",
        "text": (
            "Hydro power turns flowing water into electricity at dams "
            "and river stations. Water held behind a dam flows through "
            "tunnels and spins turbines, and generators change the "
            "spin into electricity. Hydro stations can raise or lower "
            "output quickly by opening or closing gates that control "
            "the water flow. Large dams also store water for drinking, "
            "farming, and flood control."
        ),
    },
    {
        "id": "northlight-loans",
        "topic": "library",
        "text": (
            "The lending desk on the ground floor of Northlight "
            "handles loans, returns, and reservations. Members can borrow "
            "up to ten books for three weeks and renew each loan once "
            "online. Borrowed items returned late cost a small fee "
            "for each day they are overdue. Members borrow items at "
            "the lending desk on the ground floor or at the self "
            "service machines near the entrance."
        ),
    },
    {
        "id": "northlight-hours",
        "topic": "library",
        "text": (
            "The Northlight library opens at eight in the morning on "
            "weekdays and closes at nine in the evening. On Saturdays "
            "the library opens at nine and closes at five, and on "
            "Sundays it stays closed all day. The library offers free "
            "wifi, printing, and a quiet study room on the second "
            "floor. Holiday hours are posted on the library door and "
            "the library website each month."
        ),
    },
    {
        "id": "northlight-events",
        "topic": "library",
        "text": (
            "The Northlight library runs events for children and "
            "adults every week. Storytime for young children happens "
            "on Tuesday mornings in the childrens room and lasts "
            "forty minutes. A book club for adults meets on the first "
            "Thursday evening of each month to discuss one novel. "
            "Chess club meets in the library on Friday afternoons, "
            "and boards and clocks are provided free."
        ),
    },
    {
        "id": "storms",
        "topic": "weather",
        "text": (
            "A storm is a period of strong wind, heavy rain, and "
            "sometimes thunder and lightning. Thunderstorms form when "
            "warm wet air rises quickly and cools into tall clouds. "
            "Lightning is a giant spark of electricity between a cloud "
            "and the ground, and thunder is the sound the heated air "
            "makes. Storms can flood streets, knock down trees, and "
            "cut electric power to homes."
        ),
    },
    {
        "id": "seasons",
        "topic": "weather",
        "text": (
            "Seasons are the yearly cycle of spring, summer, autumn, "
            "and winter. Seasons happen because the axis of the earth "
            "is tilted as the earth travels around the sun. When the "
            "northern half of the earth tilts toward the sun it gets "
            "summer while the southern half gets winter. Seasons "
            "change daylight length, temperature, and the growth of "
            "plants through the year."
        ),
    },
]
