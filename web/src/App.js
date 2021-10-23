import logo from "./logo.svg";
import "./App.css";
import Layout from "./components/Layout/Layout";
import theme from "./components/Layout/theme";
import { ThemeProvider } from "@mui/material";
import MapContainer from "./components/MapContainer/MapContainer";
import {useState} from "react";

function App() {
    const [data, setData] = useState({category: "Usługi reklamowe",
        center: [17.012272577777775, 51.0426686],
        city: "Wrocław",
        location: null})
  return (
    <ThemeProvider theme={theme}>
      <Layout handleDataChange={setData}>
        <MapContainer data={data}/>
      </Layout>
    </ThemeProvider>
  );
}

export default App;
