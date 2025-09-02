import {
  Box,
  Typography,
  Paper,
  Stack,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
} from "@mui/material";
import { Line } from "react-chartjs-2";
import { useState , useEffect } from "react";
import SalesMap from "./map";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);


async function fetchDashboardData() {
  const response = await fetch("http://localhost:5000/dashboard");
  if (!response.ok) {
    throw new Error("Failed to fetch dashboard data");
  }
  return response.json();
}

function DashboardPage() {
  const [dashboardData, setDashboardData] = useState(null);
  const [monthlyTrend, setMonthlyTrend] = useState({ labels: [], datasets: [] });
  useEffect(() => {
    fetchDashboardData()
      .then((data) => setDashboardData(data))
      .catch((error) => console.error(error));
  }, []);

  if (!dashboardData) {
    return <Typography>Loading...</Typography>;
  }

  const options = {
    responsive: true,
    plugins: {
      legend: { position: "top" as const },
      title: { display: true, text: "Monthly Revenue Trend" }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };

  return (
    <Box>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={2}
        mt={2}
        flexWrap="wrap"
      >
        {[
          { label: "Total Revenue", value: dashboardData["Total Revenue"], trend: "+8.2%" },
          { label: "Total Customers", value: dashboardData["Total Customers"], trend: "+12.5%" },
          { label: "Total Invoices", value: dashboardData["Total Invoices"], trend: "-2.1%" },
          // { label: "Monthly Growth", value: dashboardData["Monthly Growth"], trend: "-15.3%" },
        ].map((stat, idx) => (
          <Paper
            key={idx}
            sx={{
              p: 2,
              bgcolor: "background.paper",
              flex: { xs: "1 1 100%", sm: "1 1 calc(25% - 16px)" },
            }}
          >
            <Typography variant="h6">{stat.label}</Typography>
            <Typography variant="h5" fontWeight="bold">
              {stat.value}
            </Typography>
            <Typography
              variant="body2"
              color={stat.trend.startsWith("-") ? "error.main" : "success.main"}
            >
              {stat.trend} from last month
            </Typography>
          </Paper>
        ))}
      </Stack>

      <Stack
        direction={{ xs: "column", md: "row" }}
        spacing={2}
        mt={2}
        alignItems="stretch"
      >
        
        <Line data={{
      labels: Object.keys(dashboardData["Monthly Trend"]),          
          datasets: [
            {
              label: "Revenue",
              data: Object.values(dashboardData["Monthly Trend"]), 
              borderColor: "rgb(75, 192, 192)",
              backgroundColor: "rgba(75, 192, 192, 0.2)",
              tension: 0.4                     
            }
          ]}
        } options={options} />
      </Stack>
      <SalesMap/>
    </Box>
  );
}

export default DashboardPage;
