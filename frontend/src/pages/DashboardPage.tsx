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

function DashboardPage() {
  return (
    <Box>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={2}
        mt={2}
        flexWrap="wrap"
      >
        {[
          { label: "Total Users", value: "2,845", trend: "+12.5%" },
          { label: "Messages", value: "18,247", trend: "+8.2%" },
          { label: "Active Sessions", value: "542", trend: "-2.1%" },
          { label: "Response Time", value: "1.2s", trend: "-15.3%" },
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

      {/* Activity + Recent */}
      <Stack
        direction={{ xs: "column", md: "row" }}
        spacing={2}
        mt={2}
        alignItems="stretch"
      >
        {/* Left: Activity */}
        <Paper sx={{ p: 2, flex: 2 }}>
          <Typography variant="h6" gutterBottom>
            Activity Overview
          </Typography>
          <Typography>Chat Sessions</Typography>
          <LinearProgress variant="determinate" value={85} sx={{ mb: 2 }} />
          <Typography>User Satisfaction</Typography>
          <LinearProgress variant="determinate" value={92} sx={{ mb: 2 }} />
          <Typography>Response Rate</Typography>
          <LinearProgress variant="determinate" value={68} />
        </Paper>

        {/* Right: Recent Activity */}
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="h6" gutterBottom>
            Recent Activity
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="Alice Johnson"
                secondary="Started new chat · 2 min ago"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Bob Smith"
                secondary="Completed session · 5 min ago"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Carol Davis"
                secondary="Updated profile · 12 min ago"
              />
            </ListItem>
          </List>
        </Paper>
      </Stack>
    </Box>
  );
}

export default DashboardPage;
