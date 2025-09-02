import React, { useEffect, useState } from "react";
import {
  ComposableMap,
  Geographies,
  Geography
} from "react-simple-maps";

export default function SalesMap() {
  const [data, setData] = useState<{ [key: string]: number }>({});

  useEffect(() => {
    fetch("http://127.0.0.1:5000/dashboard")
      .then((res) => res.json())
      .then((json) => setData(json["Country Sales"]));
  }, []);

  return (
    <ComposableMap>
      <Geographies geography="/features.json">
        {({ geographies }) =>
          geographies.map((geo) => {
            const countryName = geo.properties.name;
            const value = data[countryName];

            return (
              <Geography
                key={geo.rsmKey}
                geography={geo}
                fill={value ? "rgba(4, 255, 0, 1)" : "#DDD"}
                stroke="#FFF"
              />
            );
          })
        }
      </Geographies>
    </ComposableMap>
  );
}
