
1) I want to know the list of our customers and their spending.

select b.name, sum(sellingprice) from public.transaction a
inner join public.customer b on (b.customer_id=a.customer_id)
where a.voided !='Y'
group by b.name
order by 2 desc

2) I want to find out the top 3 car manufacturers that customers bought by sales (quantity) and the sales number for it in the current month.

select * from (
select c.name, a.order_no, count(a.order_no) from public.transaction a
inner join public.car b on (b.car_id=a.car_id)
inner join public.car_manufacturer c on (c.car_manufacturer_id=b.car_manufacturer_id)
where a.voided !='Y' and transact_dt>=date_trunc('month',CURRENT_DATE)
group by c.name, a.order_no) z
order by 3 desc
limit 3

Note : May need to consider to add index for transact_dt and voided fields.

