alter table stocknews_newscontent alter column  url     drop not null;
alter table stocknews_newscontent alter column  title   drop not null;
alter table stocknews_newscontent alter column  content drop not null;
alter table stocknews_newscontent alter column  date    drop not null;

alter table stocknews_yahoocalendar alter column datetime drop not null; 
alter table stocknews_yahoocalendar alter column statistic drop not null; 
alter table stocknews_yahoocalendar alter column for_period drop not null; 
alter table stocknews_yahoocalendar alter column actual drop not null; 
alter table stocknews_yahoocalendar alter column briefing_forecast drop not null; 
alter table stocknews_yahoocalendar alter column market_expects drop not null; 
alter table stocknews_yahoocalendar alter column prior drop not null; 
alter table stocknews_yahoocalendar alter column revised_from drop not null; 
