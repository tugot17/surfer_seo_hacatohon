import * as React from "react";
import GoogleMapReact from "google-map-react";
import BusinessPoint from "./BusinessPoint";

const AnyReactComponent = ({ text }) => <div>{text}</div>;

const defaultProps = {
  center: {
    lat: 51.1263645,
    lng: 16.99177925,
  },
  zoom: 12,
};

const MapContainer = ({ data } = defaultProps) => {
  const center = {
    lat: data.center[1],
    lng: data.center[0],
  };
  const dataPoints =
    data.location != null && Array.isArray(data.location)
      ? [data.location]
      : [];
  return (
    <GoogleMapReact
      bootstrapURLKeys={{ key: "AIzaSyDoE8vCQfU9fiXudxbm1cvmt_xU4TkQESU" }}
      defaultCenter={center}
      defaultZoom={defaultProps.zoom}
    >
      {dataPoints.map(([lng, lat]) => (
        <BusinessPoint
          key={String(lat) + String(lng)}
          lat={lat}
          lng={lng}
          name={data.category}
        />
      ))}
    </GoogleMapReact>
  );
};

export default MapContainer;
